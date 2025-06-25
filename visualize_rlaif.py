"""
Visualization tools for RLAIF training process.
Creates plots and diagrams to understand the training dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Optional
import seaborn as sns

plt.style.use('seaborn-v0_8')


def plot_training_progress(results_path: str, save_path: Optional[str] = None):
    """
    Plot the training progress from RLAIF results.
    
    Args:
        results_path: Path to the results JSON file
        save_path: Optional path to save the plot
    """
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RLAIF Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Model Accuracies Comparison
    ax1 = axes[0, 0]
    models = ['Weak', 'Strong (Baseline)', 'RLAIF-Trained']
    accuracies = [results['weak_acc'], results['strong_acc'], results['rl_acc']]
    colors = ['#ff7f7f', '#7f7fff', '#7fff7f']
    
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Reward Model Loss
    ax2 = axes[0, 1]
    if 'reward_losses' in results and results['reward_losses']:
        epochs = range(1, len(results['reward_losses']) + 1)
        ax2.plot(epochs, results['reward_losses'], 'o-', color='red', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Reward Model Loss')
        ax2.set_title('Reward Model Training Loss')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No reward loss data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Reward Model Training Loss')
    
    # Plot 3: RL Policy Loss
    ax3 = axes[1, 0]
    if 'rl_losses' in results and results['rl_losses']:
        epochs = range(1, len(results['rl_losses']) + 1)
        ax3.plot(epochs, results['rl_losses'], 's-', color='blue', linewidth=2, markersize=6)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Policy Loss')
        ax3.set_title('RL Policy Training Loss')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No policy loss data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('RL Policy Training Loss')
    
    # Plot 4: Training Configuration
    ax4 = axes[1, 1]
    ax4.axis('off')
    config = results.get('config', {})
    
    config_text = f"""
    Configuration:
    
    Dataset: {config.get('dataset', 'N/A')}
    Weak Model: {config.get('weak_model_size', 'N/A')}
    Strong Model: {config.get('strong_model_size', 'N/A')}
    RL Epochs: {config.get('rl_epochs', 'N/A')}
    Preference Pairs: {results.get('n_preference_pairs', 'N/A')}
    Reward Hidden Size: {config.get('reward_model_hidden_size', 'N/A')}
    
    Results:
    RLAIF Improvement: {results['rl_acc'] - results['strong_acc']:+.4f}
    """
    
    ax4.text(0.1, 0.9, config_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def create_architecture_diagram(save_path: Optional[str] = None):
    """
    Create a diagram showing the RLAIF architecture.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'RLAIF Architecture for Deception Mitigation', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Stage 1: Ground Truth Training
    rect1 = plt.Rectangle((0.5, 5.5), 2, 1.5, facecolor='lightblue', 
                         edgecolor='blue', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.5, 6.25, 'Stage 1:\nGround Truth\nTraining', ha='center', va='center', 
            fontweight='bold')
    
    # Stage 2: Preference Generation
    rect2 = plt.Rectangle((3.5, 5.5), 2, 1.5, facecolor='lightgreen', 
                         edgecolor='green', linewidth=2)
    ax.add_patch(rect2)
    ax.text(4.5, 6.25, 'Stage 2:\nPreference\nGeneration', ha='center', va='center', 
            fontweight='bold')
    
    # Stage 3: RLAIF Training
    rect3 = plt.Rectangle((6.5, 5.5), 2, 1.5, facecolor='lightyellow', 
                         edgecolor='orange', linewidth=2)
    ax.add_patch(rect3)
    ax.text(7.5, 6.25, 'Stage 3:\nRLAIF\nTraining', ha='center', va='center', 
            fontweight='bold')
    
    # Data splits
    ax.text(1.5, 4.8, 'Ground Truth\nData (40%)', ha='center', va='center', 
            fontsize=10, style='italic')
    ax.text(4.5, 4.8, 'Preference\nData (40%)', ha='center', va='center', 
            fontsize=10, style='italic')
    ax.text(7.5, 4.8, 'Test Data\n(20%)', ha='center', va='center', 
            fontsize=10, style='italic')
    
    # Models
    ax.text(1, 3.8, 'Weak Model', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))
    ax.text(2, 3.8, 'Strong Model', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Components
    ax.text(4.5, 3.8, 'Preference Pairs\n(Chosen/Rejected)', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.text(6.5, 3.8, 'RNN Reward\nModel', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(8.5, 3.8, 'PPO Trainer', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Arrows
    ax.annotate('', xy=(3.3, 6.25), xytext=(2.7, 6.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(6.3, 6.25), xytext=(5.7, 6.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Key features
    features_text = """
    Key Features:
    • Three-way dataset splitting
    • RNN-based reward modeling
    • Trajectory-aware learning
    • Deception detection
    • Partial observability handling
    • PPO-based policy optimization
    """
    
    ax.text(1, 2.5, features_text, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    # Deception mitigation
    mitigation_text = """
    Deception Mitigation:
    • Confidence-inconsistency detection
    • Historical pattern analysis
    • Adaptive reward weighting
    • Robust preference learning
    • Online supervision quality assessment
    """
    
    ax.text(6, 2.5, mitigation_text, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to: {save_path}")
    else:
        plt.show()


def plot_preference_analysis(preference_data: List[Dict], save_path: Optional[str] = None):
    """
    Analyze and visualize preference data quality.
    """
    if not preference_data:
        print("No preference data provided")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Preference Data Analysis', fontsize=14, fontweight='bold')
    
    # Extract data
    confidences = [p['weak_confidence'] for p in preference_data]
    ground_truths = [p['ground_truth'] for p in preference_data]
    weak_preds = [p['weak_prediction'] for p in preference_data]
    
    # Plot 1: Confidence distribution
    ax1 = axes[0, 0]
    ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Weak Model Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Distribution')
    ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    ax1.legend()
    
    # Plot 2: Agreement analysis
    ax2 = axes[0, 1]
    agreements = [1 if gt == wp else 0 for gt, wp in zip(ground_truths, weak_preds)]
    agreement_rate = np.mean(agreements)
    
    labels = ['Disagree', 'Agree']
    counts = [len(agreements) - sum(agreements), sum(agreements)]
    colors = ['lightcoral', 'lightgreen']
    
    ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Weak-Ground Truth Agreement\n(Rate: {agreement_rate:.3f})')
    
    # Plot 3: Confidence vs Agreement
    ax3 = axes[1, 0]
    agree_conf = [c for c, a in zip(confidences, agreements) if a == 1]
    disagree_conf = [c for c, a in zip(confidences, agreements) if a == 0]
    
    ax3.hist(agree_conf, bins=15, alpha=0.7, label='Agree', color='green')
    ax3.hist(disagree_conf, bins=15, alpha=0.7, label='Disagree', color='red')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Confidence by Agreement')
    ax3.legend()
    
    # Plot 4: Deception potential
    ax4 = axes[1, 1]
    # High confidence disagreements are potential deception
    deception_scores = [c if a == 0 else 0 for c, a in zip(confidences, agreements)]
    high_conf_disagree = sum(1 for c, a in zip(confidences, agreements) 
                           if c > 0.8 and a == 0)
    
    ax4.bar(['Low Risk', 'Medium Risk', 'High Risk'], 
            [len(preference_data) - high_conf_disagree - len([d for d in deception_scores if 0.5 < d <= 0.8]),
             len([d for d in deception_scores if 0.5 < d <= 0.8]),
             high_conf_disagree],
            color=['green', 'yellow', 'red'], alpha=0.7)
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Deception Risk Assessment')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Preference analysis saved to: {save_path}")
    else:
        plt.show()


def main():
    """Main function for testing visualization tools."""
    # Create sample data for testing
    import tempfile
    
    # Sample results data
    sample_results = {
        "weak_acc": 0.75,
        "strong_acc": 0.85,
        "rl_acc": 0.88,
        "reward_losses": [0.8, 0.6, 0.4],
        "rl_losses": [0.5, 0.3, 0.2, 0.15, 0.12],
        "n_preference_pairs": 500,
        "config": {
            "weak_model_size": "gpt2",
            "strong_model_size": "gpt2-xl",
            "dataset": "sciq",
            "rl_epochs": 5,
            "reward_model_hidden_size": 128
        }
    }
    
    # Save sample results
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_results, f)
        results_path = f.name
    
    # Test visualizations
    print("Creating training progress plot...")
    plot_training_progress(results_path)
    
    print("Creating architecture diagram...")
    create_architecture_diagram()
    
    # Clean up
    os.unlink(results_path)


if __name__ == "__main__":
    main()
