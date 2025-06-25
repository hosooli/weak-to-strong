import json
import os
from typing import Dict, List, Optional, Sequence, Union, Tuple
import random

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer, clear_mem
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model
from weak_to_strong.model import TransformerWithHead
from weak_to_strong.eval import eval_model_acc

# Import MODEL_CONFIGS from original script
from train_weak_to_strong import MODEL_CONFIGS, MODELS_DICT

class RNNRewardModel(nn.Module):
    """
    RNN-based reward model that takes into account the history of per-token log probabilities
    to model partial observability and deception patterns.
    """
    def __init__(self, hidden_size: int = 128, num_layers: int = 2, history_length: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.history_length = history_length
        
        # Input: per-token log probabilities + trajectory features
        self.input_projection = nn.Linear(4, hidden_size)  # [log_prob_chosen, log_prob_rejected, length_diff, confidence_diff]
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, trajectory_features: torch.Tensor, hidden_state=None):
        """
        Args:
            trajectory_features: [batch_size, sequence_length, 4] tensor containing:
                - log_prob_chosen: log probability of chosen response
                - log_prob_rejected: log probability of rejected response 
                - length_diff: length difference between responses
                - confidence_diff: confidence difference between responses
            hidden_state: previous hidden state for online learning
        """
        batch_size, seq_len, _ = trajectory_features.shape
        
        # Project input features
        x = self.input_projection(trajectory_features)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Pass through RNN
        if hidden_state is not None:
            rnn_out, new_hidden = self.rnn(x, hidden_state)
        else:
            rnn_out, new_hidden = self.rnn(x)
        
        # Use the last output for reward prediction
        reward = self.reward_head(rnn_out[:, -1, :])
        
        return reward.squeeze(-1), new_hidden

class PPOTrainer:
    """
    PPO trainer for online weak-to-strong alignment with preference learning.
    """
    def __init__(self, 
                 policy_model: TransformerWithHead,
                 reward_model: RNNRewardModel,
                 ref_model: TransformerWithHead,
                 lr: float = 1e-5,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        
        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
        self.reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Value function (can be separate network or shared)
        hidden_size = getattr(policy_model.config, "n_embd", getattr(policy_model.config, "hidden_size", 768))
        self.value_head = nn.Linear(hidden_size, 1).to(policy_model.device if hasattr(policy_model, 'device') else 'cuda')
        
    def compute_trajectory_features(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor):
        """
        Compute trajectory features for the reward model.
        """
        with torch.no_grad():
            chosen_logits = self.policy_model(chosen_ids)
            rejected_logits = self.policy_model(rejected_ids)
            
            chosen_probs = F.softmax(chosen_logits, dim=-1)
            rejected_probs = F.softmax(rejected_logits, dim=-1)
            
            # Get log probabilities for the preferred choice (index 1)
            log_prob_chosen = torch.log(chosen_probs[:, 1] + 1e-8)
            log_prob_rejected = torch.log(rejected_probs[:, 1] + 1e-8)
            
            # Compute additional features
            length_diff = torch.tensor([len(c) - len(r) for c, r in zip(chosen_ids, rejected_ids)], 
                                     dtype=torch.float32, device=chosen_ids.device)
            confidence_diff = torch.max(chosen_probs, dim=-1)[0] - torch.max(rejected_probs, dim=-1)[0]
            
            # Stack features
            features = torch.stack([log_prob_chosen, log_prob_rejected, length_diff, confidence_diff], dim=-1)
            
        return features.unsqueeze(1)  # Add sequence dimension
    
    def update_reward_model(self, preference_batch):
        """
        Update the reward model based on preference data.
        """
        chosen_ids = preference_batch['chosen_ids']
        rejected_ids = preference_batch['rejected_ids']
        
        # Compute trajectory features
        trajectory_features = self.compute_trajectory_features(chosen_ids, rejected_ids)
        
        # Get reward predictions
        chosen_rewards, _ = self.reward_model(trajectory_features)
        rejected_rewards, _ = self.reward_model(trajectory_features)
        
        # Bradley-Terry preference loss
        preference_loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards) + 1e-8).mean()
        
        self.reward_optimizer.zero_grad()
        preference_loss.backward()
        self.reward_optimizer.step()
        
        return preference_loss.item()
    
    def ppo_update(self, batch):
        """
        Perform PPO update on the policy model.
        """
        input_ids = batch['input_ids']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # Forward pass
        logits = self.policy_model(input_ids)
        dist = Categorical(logits=logits)
        
        # Get new log probabilities
        actions = torch.argmax(logits, dim=-1)
        new_log_probs = dist.log_prob(actions)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value_head(self.policy_model.transformer(input_ids)[0][:, -1, :])
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy loss
        entropy_loss = -dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }

def create_preference_pairs(weak_model, dataset, tokenizer, max_ctx: int, n_pairs: int = 1000):
    """
    Use the weak model to create preference pairs (chosen/rejected) from the dataset.
    """
    preference_pairs = []
    
    weak_model.eval()
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= n_pairs:
                break
                
            input_text = example['txt']
            true_label = example['hard_label']
            
            # Generate two responses: one aligned with weak model preference, one not
            tokens = tokenizer(input_text)['input_ids']
            if len(tokens) >= max_ctx:
                continue
                
            input_ids = torch.tensor([tokens]).to(weak_model.device if hasattr(weak_model, 'device') else 'cuda')
            logits = weak_model(input_ids)
            probs = F.softmax(logits, dim=-1)
            
            # Create preference based on weak model confidence
            weak_confidence = torch.max(probs, dim=-1)[0].item()
            weak_pred = torch.argmax(probs, dim=-1).item()
            
            # Choose preferred response based on weak model prediction
            if weak_pred == true_label:
                # Weak model agrees with ground truth
                chosen_label = true_label
                rejected_label = 1 - true_label
            else:
                # Weak model disagrees - create deceptive scenario
                chosen_label = weak_pred  # What weak model prefers
                rejected_label = true_label  # Ground truth (rejected by weak model)
            
            # Create synthetic responses (in practice, these would be generated)
            chosen_response = f"{input_text} [CHOSEN: {chosen_label}]"
            rejected_response = f"{input_text} [REJECTED: {rejected_label}]"
            
            chosen_tokens = tokenizer(chosen_response)['input_ids']
            rejected_tokens = tokenizer(rejected_response)['input_ids']
            
            if len(chosen_tokens) < max_ctx and len(rejected_tokens) < max_ctx:
                preference_pairs.append({
                    'chosen_ids': torch.tensor(chosen_tokens),
                    'rejected_ids': torch.tensor(rejected_tokens),
                    'chosen_text': chosen_response,
                    'rejected_text': rejected_response,
                    'weak_confidence': weak_confidence,
                    'ground_truth': true_label,
                    'weak_prediction': weak_pred
                })
    
    return preference_pairs

def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq", 
    n_docs: int = 15000,  # Increased for 3-way split
    n_test_docs: int = 200,
    weak_model_size: str = "gpt2",
    strong_model_size: str = "gpt2-xl",
    weak_lr: Optional[float] = None,
    strong_lr: Optional[float] = None,
    rl_lr: float = 1e-5,
    gt_epochs: int = 2,
    rl_epochs: int = 5,
    n_preference_pairs: int = 1000,
    reward_model_hidden_size: int = 128,
    reward_model_layers: int = 2,
    history_length: int = 10,
    ppo_epsilon: float = 0.2,
    gamma: float = 0.99,
    force_retrain: bool = False,
    seed: int = 0,
    results_folder: str = "/tmp/results_rlaif",
    eval_every: int = 100,
):
    """
    Main function for RLAIF-based weak-to-strong training.
    
    This implements the three-stage process:
    1. Train weak and strong models on ground truth data (Section 1)
    2. Create preference pairs using weak model on held-out data (Section 2) 
    3. Train RL policy to align strong model using RNN-based reward model (Section 3)
    """
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Setup model configs
    assert weak_model_size in MODELS_DICT, f"Unknown model size {weak_model_size}"
    assert strong_model_size in MODELS_DICT, f"Unknown model size {strong_model_size}"
    
    weak_model_config = MODELS_DICT[weak_model_size]
    strong_model_config = MODELS_DICT[strong_model_size]
    
    if weak_lr is None:
        weak_lr = weak_model_config.default_lr
    if strong_lr is None:
        strong_lr = strong_model_config.default_lr
    
    print(f"Starting RLAIF training with {weak_model_size} -> {strong_model_size}")
    
    # Load and split dataset into 3 sections
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))
    train_dataset, test_ds = dataset["train"], dataset["test"]
    
    # Three-way split: ground_truth (40%), preference (40%), held_out (20%)
    split1 = train_dataset.train_test_split(test_size=0.6, seed=seed)  # 40% / 60%
    ground_truth_ds = split1["train"]
    remaining_ds = split1["test"]
    
    split2 = remaining_ds.train_test_split(test_size=0.33, seed=seed)  # 40% / 20% of total
    preference_ds = split2["train"] 
    held_out_ds = split2["test"]
    
    print(f"Dataset splits - Ground truth: {len(ground_truth_ds)}, Preference: {len(preference_ds)}, Held-out: {len(held_out_ds)}, Test: {len(test_ds)}")
    
    # Stage 1: Train weak and strong models on ground truth data
    tokenizer = get_tokenizer(weak_model_config.name)
    
    def train_baseline_model(model_config, train_ds, label):
        print(f"Training {label} model on ground truth data")
        
        # Tokenize dataset
        tokenized_train = tokenize_dataset(train_ds, tokenizer, max_ctx)
        tokenized_test = tokenize_dataset(test_ds, tokenizer, max_ctx)
        
        save_path = os.path.join(results_folder, f"{label}_model_gt", model_config.name.replace("/", "_"))
        
        test_results, _ = train_and_save_model(
            model_config,
            tokenized_train,
            tokenized_test,
            batch_size=batch_size,
            save_path=save_path,
            loss_fn=xent_loss(),
            lr=weak_lr if label == "weak" else strong_lr,
            epochs=gt_epochs,
            force_retrain=force_retrain,
            eval_batch_size=model_config.eval_batch_size,
            eval_every=eval_every,
        )
        
        # Load the trained model
        if model_config.model_parallel:
            model = TransformerWithHead.from_pretrained(
                model_config.name,
                num_labels=2,
                device_map="auto",
                **model_config.custom_kwargs or {}
            )
        else:
            model = TransformerWithHead.from_pretrained(
                model_config.name, 
                num_labels=2,
                **model_config.custom_kwargs or {}
            ).to("cuda")
        
        # Load trained weights
        if os.path.exists(os.path.join(save_path, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
            model.load_state_dict(state_dict, strict=False)
        
        return model, test_results
    
    # Train baseline models
    weak_model, weak_results = train_baseline_model(weak_model_config, ground_truth_ds, "weak")
    strong_model, strong_results = train_baseline_model(strong_model_config, ground_truth_ds, "strong")
    
    # Stage 2: Create preference pairs using weak model
    print("Creating preference pairs using weak model")
    preference_pairs = create_preference_pairs(
        weak_model, preference_ds, tokenizer, max_ctx, n_preference_pairs
    )
    print(f"Created {len(preference_pairs)} preference pairs")
    
    # Stage 3: RLAIF training
    print("Starting RLAIF training")
    
    # Initialize components for RL training
    reward_model = RNNRewardModel(
        hidden_size=reward_model_hidden_size,
        num_layers=reward_model_layers, 
        history_length=history_length
    ).to("cuda")
    
    # Create reference model (copy of strong model)
    ref_model = strong_model  # In practice, would create a separate copy
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        policy_model=strong_model,
        reward_model=reward_model,
        ref_model=ref_model,
        lr=rl_lr,
        gamma=gamma,
        epsilon=ppo_epsilon
    )
    
    # Train reward model on preference data
    print("Training reward model on preferences")
    reward_losses = []
    for epoch in range(3):  # Few epochs for reward model
        epoch_losses = []
        for i in range(0, len(preference_pairs), batch_size):
            batch_pairs = preference_pairs[i:i+batch_size]
            
            # Create batch
            chosen_ids = torch.stack([pair['chosen_ids'] for pair in batch_pairs]).to("cuda")
            rejected_ids = torch.stack([pair['rejected_ids'] for pair in batch_pairs]).to("cuda")
            
            batch = {
                'chosen_ids': chosen_ids,
                'rejected_ids': rejected_ids
            }
            
            loss = ppo_trainer.update_reward_model(batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        reward_losses.append(avg_loss)
        print(f"Reward model epoch {epoch+1}, avg loss: {avg_loss:.4f}")
    
    # RL training loop
    print("Starting RL policy training")
    rl_losses = []
    
    for epoch in range(rl_epochs):
        epoch_policy_losses = []
        
        # Sample from preference data for RL training
        for i in range(0, min(len(preference_pairs), 500), batch_size):  # Limit for efficiency
            batch_pairs = preference_pairs[i:i+batch_size]
            
            # Create training batch (simplified)
            input_ids = torch.stack([pair['chosen_ids'] for pair in batch_pairs]).to("cuda")
            
            # Get old policy log probs
            with torch.no_grad():
                old_logits = strong_model(input_ids)
                old_dist = Categorical(logits=old_logits)
                old_actions = torch.argmax(old_logits, dim=-1)
                old_log_probs = old_dist.log_prob(old_actions)
                
                # Compute rewards using reward model
                trajectory_features = ppo_trainer.compute_trajectory_features(input_ids, input_ids)
                rewards, _ = reward_model(trajectory_features)
                
                # Simple advantage estimation (in practice would use GAE)
                advantages = rewards - rewards.mean()
                returns = rewards
            
            batch = {
                'input_ids': input_ids,
                'old_log_probs': old_log_probs,
                'advantages': advantages,
                'returns': returns
            }
            
            # PPO update
            losses = ppo_trainer.ppo_update(batch)
            epoch_policy_losses.append(losses['total_loss'])
        
        avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0.0
        rl_losses.append(avg_policy_loss)
        print(f"RL epoch {epoch+1}, avg policy loss: {avg_policy_loss:.4f}")
        
        # Evaluate periodically
        if (epoch + 1) % eval_every == 0:
            print("Evaluating RL-trained model")
            tokenized_test = tokenize_dataset(test_ds, tokenizer, max_ctx)
            rl_results = eval_model_acc(strong_model, tokenized_test, strong_model_config.eval_batch_size)
            rl_acc = np.mean([r["acc"] for r in rl_results])
            print(f"RL model accuracy: {rl_acc:.4f}")
    
    # Final evaluation
    print("Final evaluation")
    tokenized_test = tokenize_dataset(test_ds, tokenizer, max_ctx)
    
    weak_acc = np.mean([x["acc"] for x in weak_results])
    strong_acc = np.mean([x["acc"] for x in strong_results])
    
    final_rl_results = eval_model_acc(strong_model, tokenized_test, strong_model_config.eval_batch_size)
    rl_acc = np.mean([r["acc"] for r in final_rl_results])
    
    # Save results
    results = {
        "weak_acc": weak_acc,
        "strong_acc": strong_acc, 
        "rl_acc": rl_acc,
        "reward_losses": reward_losses,
        "rl_losses": rl_losses,
        "n_preference_pairs": len(preference_pairs),
        "config": {
            "weak_model_size": weak_model_size,
            "strong_model_size": strong_model_size,
            "dataset": ds_name,
            "rl_epochs": rl_epochs,
            "reward_model_hidden_size": reward_model_hidden_size,
        }
    }
    
    results_path = os.path.join(results_folder, f"rlaif_results_{weak_model_size}_{strong_model_size}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"Weak model accuracy: {weak_acc:.4f}")
    print(f"Strong model (baseline) accuracy: {strong_acc:.4f}")
    print(f"RLAIF-trained model accuracy: {rl_acc:.4f}")
    print(f"Results saved to: {results_path}")
    
    # Cleanup
    clear_mem()
    
    return results

if __name__ == "__main__":
    fire.Fire(main) 