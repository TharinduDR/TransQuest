#!/usr/bin/env python3
"""
Test script to compare simplified hyperbolic network vs original.
"""

import os
import sys
import pandas as pd
import torch
import logging

# Add TransQuest to path
sys.path.append('.')

from transquest.algo.sentence_level.hypertransquest.simple_hyperbolic_net import FixedHyperbolicSiameseNet
from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_model():
    """Test the simplified hyperbolic model"""
    logger.info("Testing Simplified Hyperbolic Model...")
    
    # Test parameters
    model_args = {
        "num_train_epochs": 2,  # Very short test
        "learning_rate": 2e-5,
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "max_seq_length": 128,
        "evaluate_during_training_steps": 0,
        "manual_seed": 42,
        "best_model_dir": "simple_test_outputs",
        "output_dir": "simple_test_outputs_checkpoints"
    }
    
    # Create sample data
    sample_data = [
        ("Hello world", "Bonjour monde", 0.9),  # High quality
        ("Good morning", "Bonjour", 0.8),       # High quality
        ("How are you?", "Comment ça va?", 0.7),  # Medium quality
        ("Test sentence", "Pharmaceuticals are dangerous", 0.2)  # Low quality
    ]
    
    train_df = pd.DataFrame(sample_data, columns=['text_a', 'text_b', 'labels'])
    eval_df = train_df  # Use same for eval in test
    
    try:
        # Initialize simplified model
        model = HyperTransQuestModel(
            model_name="xlm-roberta-base",
            args=model_args,
            curvature=1.0,
            dropout=0.2
        )
        
        logger.info("✅ Simplified model initialized successfully")
        
        # Test parameter groups
        param_groups = model._create_parameter_groups()
        logger.info(f"✅ Parameter groups: {len(param_groups)} groups")
        
        # Test forward pass
        test_samples = [
            ("Hello world", "Bonjour monde"),
            ("Good morning", "Buenos días"),
        ]
        
        scores = model.predict(test_samples, verbose=True)
        logger.info(f"✅ Test predictions: {scores}")
        
        # Test training step (single batch)
        logger.info("Testing single training step...")
        model.train_model(train_df, eval_df, verbose=True)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simplified model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info("Simplified Hyperbolic Model Test")
    logger.info("="*50)
    
    success = test_simple_model()
    
    if success:
        logger.info("🎉 Simplified model test passed!")
        logger.info("Ready to compare performance with original implementation")
    else:
        logger.error("❌ Simplified model test failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)