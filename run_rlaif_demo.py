#!/usr/bin/env python3
"""
Demo script for running RLAIF-based weak-to-strong training.

This script demonstrates the three-stage process:
1. Train weak and strong models on ground truth data
2. Generate preference pairs using the weak model
3. Train the strong model using RLAIF with RNN-based reward modeling

Usage:
    python run_rlaif_demo.py --ds_name sciq --weak_model_size gpt2 --strong_model_size gpt2-medium
"""

import os
import argparse
from train_weak_to_strong_rlaif import main as rlaif_main


def run_demo():
    """Run a demonstration of the RLAIF training process."""
    
    # Set up demo parameters
    demo_config = {
        "batch_size": 16,  # Smaller for demo
        "max_ctx": 512,    # Smaller context for faster training
        "ds_name": "sciq",
        "n_docs": 2000,    # Reduced dataset size for demo
        "n_test_docs": 100,
        "weak_model_size": "gpt2",
        "strong_model_size": "gpt2-medium",  # Smaller strong model for demo
        "gt_epochs": 1,    # Fewer epochs for demo
        "rl_epochs": 3,
        "n_preference_pairs": 200,  # Fewer pairs for demo
        "reward_model_hidden_size": 64,
        "reward_model_layers": 1,
        "history_length": 5,
        "seed": 42,
        "results_folder": "./demo_results_rlaif",
        "force_retrain": True,  # Always retrain for demo
        "eval_every": 50,
    }
    
    print("🚀 Starting RLAIF Demo")
    print("=" * 50)
    print(f"Dataset: {demo_config['ds_name']}")
    print(f"Weak model: {demo_config['weak_model_size']}")
    print(f"Strong model: {demo_config['strong_model_size']}")
    print(f"Training documents: {demo_config['n_docs']}")
    print(f"Test documents: {demo_config['n_test_docs']}")
    print("=" * 50)
    
    # Create results directory
    os.makedirs(demo_config["results_folder"], exist_ok=True)
    
    try:
        # Run RLAIF training
        results = rlaif_main(**demo_config)
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 50)
        print("RESULTS SUMMARY:")
        print(f"Weak model accuracy: {results['weak_acc']:.4f}")
        print(f"Strong model (baseline) accuracy: {results['strong_acc']:.4f}")
        print(f"RLAIF-trained model accuracy: {results['rl_acc']:.4f}")
        
        # Calculate improvement
        improvement = results['rl_acc'] - results['strong_acc']
        print(f"RLAIF improvement: {improvement:+.4f}")
        
        print(f"\nPreference pairs created: {results['n_preference_pairs']}")
        print(f"Results saved to: {demo_config['results_folder']}")
        
        # Success indicators
        if improvement > 0:
            print("\n✅ RLAIF training improved over baseline!")
        else:
            print("\n⚠️  RLAIF did not improve over baseline (this may be normal for demo)")
            
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This might be due to:")
        print("- Missing dependencies (torch, transformers, etc.)")
        print("- Insufficient GPU memory")
        print("- Dataset loading issues")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RLAIF demo")
    parser.add_argument("--ds_name", default="sciq", help="Dataset name")
    parser.add_argument("--weak_model_size", default="gpt2", help="Weak model size")
    parser.add_argument("--strong_model_size", default="gpt2-medium", help="Strong model size")
    parser.add_argument("--quick", action="store_true", help="Run very quick demo with minimal training")
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick demo mode (minimal training)...")
        # Override for very quick demo
        os.environ["DEMO_QUICK"] = "1"
    
    run_demo()
