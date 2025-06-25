# RLAIF: Deception Mitigation in Online Weak-to-Strong Alignment

This implementation provides a solution for **Deception Mitigation in Online Weak-to-Strong Alignment via Reinforcement Learning from AI Feedback (RLAIF) under Partial Observability**.

## Overview

The traditional weak-to-strong learning paradigm faces challenges when the weak supervisor might provide deceptive or inconsistent feedback. This implementation addresses these challenges through:

1. **Three-way dataset splitting** instead of the traditional two-way split
2. **Preference-based learning** using weak model labels
3. **RNN-based reward modeling** that accounts for trajectory history
4. **Partial observability modeling** to handle uncertainty in supervision
5. **Deception detection mechanisms** to identify potentially misleading feedback

## Architecture

### 1. Three-Stage Training Process

#### Stage 1: Ground Truth Training
- Both weak and strong models are trained on the same ground truth data
- Establishes baseline performance for both models
- Ensures fair comparison and prevents distribution shift

#### Stage 2: Preference Generation
- Weak model generates preferences (chosen/rejected pairs) on held-out data
- Creates scenarios where weak model preferences may conflict with ground truth
- Identifies potential deception patterns in weak supervision

#### Stage 3: RLAIF Training
- Strong model is trained using PPO with RNN-based reward model
- Reward model considers trajectory history for partial observability
- Online learning adapts to changing supervision quality

### 2. Key Components

#### RNN Reward Model
```python
class RNNRewardModel(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, history_length=10):
        # Features: [log_prob_chosen, log_prob_rejected, length_diff, confidence_diff]
        self.input_projection = nn.Linear(4, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.reward_head = nn.Linear(hidden_size, 1)
```

#### PPO Trainer
- Implements Proximal Policy Optimization for stable RL training
- Includes value function estimation and entropy regularization
- Handles preference-based reward signals

#### Deception Detection
- Monitors consistency of weak model predictions
- Detects confidence-inconsistency patterns
- Provides early warning for potential deceptive behavior

## Usage

### Basic Usage

```bash
# Run RLAIF training
python train_weak_to_strong_rlaif.py \
    --ds_name sciq \
    --weak_model_size gpt2 \
    --strong_model_size gpt2-xl \
    --n_docs 15000 \
    --rl_epochs 10 \
    --n_preference_pairs 2000
```

### Demo

```bash
# Run quick demo
python run_rlaif_demo.py --quick

# Run full demo with specific models
python run_rlaif_demo.py \
    --weak_model_size gpt2 \
    --strong_model_size gpt2-medium \
    --ds_name sciq
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ds_name` | Dataset name (sciq, boolq, etc.) | `"sciq"` |
| `n_docs` | Total training documents | `15000` |
| `weak_model_size` | Weak model size | `"gpt2"` |
| `strong_model_size` | Strong model size | `"gpt2-xl"` |
| `rl_epochs` | RL training epochs | `5` |
| `n_preference_pairs` | Number of preference pairs | `1000` |
| `reward_model_hidden_size` | RNN hidden size | `128` |
| `history_length` | Trajectory history length | `10` |
| `ppo_epsilon` | PPO clipping parameter | `0.2` |
| `gamma` | Discount factor | `0.99` |

## Dataset Splitting Strategy

The implementation uses a three-way split:

1. **Ground Truth (40%)**: For training both weak and strong models
2. **Preference (40%)**: For generating weak model preferences  
3. **Held-out (20%)**: Reserved for potential future use
4. **Test**: Separate test set for final evaluation

This ensures:
- No data leakage between training stages
- Sufficient data for preference generation
- Fair evaluation on unseen data

## Deception Mitigation Strategies

### 1. Trajectory-Aware Rewards
- RNN reward model considers sequence of decisions
- Detects patterns in supervision quality over time
- Adapts to changing weak model behavior

### 2. Preference Strength Modeling
- Measures confidence in preference decisions
- Weights preferences by reliability
- Reduces impact of uncertain supervision

### 3. Partial Observability Handling
- Maintains belief states about true supervision quality
- Incorporates uncertainty in reward estimation
- Provides robust learning under incomplete information

## Results and Evaluation

The implementation tracks multiple metrics:

- **Baseline Accuracies**: Weak and strong model performance on ground truth
- **RLAIF Accuracy**: Performance after RL training
- **Preference Quality**: Analysis of weak model preferences
- **Deception Signals**: Detection of problematic supervision patterns

### Expected Outcomes

1. **Improved Alignment**: RLAIF training should improve strong model alignment
2. **Deception Robustness**: System should handle inconsistent weak supervision
3. **Stable Learning**: Training should be stable despite noisy preferences

## Technical Details

### Memory Requirements
- RNN reward model: ~10MB additional parameters
- Trajectory buffer: Configurable, typically ~100MB
- PPO requires storing old policy outputs

### Computational Complexity
- Training time: ~2-3x longer than standard weak-to-strong
- Memory usage: ~1.5x baseline due to trajectory tracking
- Parallelizable across multiple GPUs

### Dependencies
```bash
pip install torch>=1.9.0
pip install transformers>=4.20.0
pip install datasets>=2.0.0
pip install numpy>=1.21.0
pip install fire>=0.4.0
```

## Future Extensions

1. **Multi-agent RLAIF**: Multiple weak supervisors
2. **Adversarial Training**: Explicit deception modeling
3. **Meta-learning**: Learning to detect deception patterns
4. **Interpretability**: Understanding learned reward functions

## Troubleshooting

### Common Issues

1. **GPU Memory**: Reduce batch size or model sizes
2. **Convergence**: Adjust learning rates or increase epochs
3. **Preference Quality**: Check weak model training quality

### Debug Mode
```bash
python train_weak_to_strong_rlaif.py --debug --eval_every 10
```

## Citation

If you use this implementation, please cite:

```bibtex
@misc{rlaif_weak_to_strong_2024,
  title={RLAIF: Deception Mitigation in Online Weak-to-Strong Alignment},
  author={Implementation based on weak-to-strong framework},
  year={2024},
  url={https://github.com/openai/weak-to-strong}
}
```

## License

This implementation extends the original weak-to-strong codebase and follows the same licensing terms.
