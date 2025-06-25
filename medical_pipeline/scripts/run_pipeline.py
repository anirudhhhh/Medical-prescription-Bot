#!/usr/bin/env python3
"""
Main pipeline script for knowledge distillation
Orchestrates the entire process from data generation to model training
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import DataGenerator
from student_trainer import StudentTrainer

def setup_environment():
    """Setup environment and check dependencies"""
    print("🔧 Setting up environment...")
    
    # Check if we're in the right directory
    if not os.path.exists("configs/model_configs.json"):
        print("❌ Error: Please run this script from the knowledge_distillation_pipeline directory")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "models", "outputs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Environment setup complete")

def check_api_keys(teacher_model):
    """Check if required API key is set"""
    config_path = "configs/model_configs.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key_env = config['teacher_models'][teacher_model]['api_key_env']
    if not os.getenv(api_key_env):
        print(f"❌ Error: {api_key_env} environment variable not set")
        print(f"Please set your API key: export {api_key_env}='your_api_key'")
        sys.exit(1)
    
    print(f"✅ API key for {teacher_model} is configured")

def run_data_generation(teacher_model, num_samples, topic=None, validate=True):
    """Run data generation phase"""
    print(f"\n📊 Starting data generation with {teacher_model}...")
    
    # Determine output path
    if topic:
        output_path = f"data/{topic}_dataset.jsonl"
    else:
        output_path = "data/generated_dataset.jsonl"
    
    # Initialize generator
    generator = DataGenerator(teacher_model)
    
    # Generate dataset
    if topic:
        generator.generate_topic_specific_dataset(topic, num_samples, output_path)
    else:
        generator.generate_dataset(output_path, num_samples)
    
    # Validate if requested
    if validate:
        print("\n🔍 Validating generated dataset...")
        stats = generator.validate_dataset(output_path)
        print(f"📈 Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Valid JSON: {stats['valid_json_count']}")
        print(f"  Average conversation length: {stats['avg_conversation_length']:.1f} exchanges")
        print(f"  Topics covered: {stats['topics_covered']}")
        
        if stats['errors']:
            print(f"  ⚠️  Errors: {len(stats['errors'])}")
    
    return output_path

def run_model_training(model_type, train_data_path, num_epochs, evaluate=True):
    """Run model training phase"""
    print(f"\n🤖 Starting model training with {model_type}...")
    
    # Determine output directory
    output_dir = f"models/{model_type}_student"
    
    # Initialize trainer
    trainer = StudentTrainer(model_type)
    
    # Train model
    model_path = trainer.train(
        train_data_path,
        output_dir,
        num_epochs
    )
    
    # Evaluate if requested
    if evaluate:
        print("\n📊 Evaluating trained model...")
        results = trainer.evaluate(train_data_path)
        print(f"📈 Evaluation Results:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Valid JSON outputs: {results['valid_json_count']}")
        
        # Save sample outputs
        sample_output_path = f"outputs/{model_type}_sample_outputs.json"
        with open(sample_output_path, 'w') as f:
            json.dump(results['sample_outputs'], f, indent=2)
        print(f"  Sample outputs saved to {sample_output_path}")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description='Medical Conversation Knowledge Distillation Pipeline')
    
    # Data generation arguments
    parser.add_argument('--teacher_model', type=str, default='deepseek',
                       choices=['deepseek', 'groq', 'openai', 'anthropic'],
                       help='Teacher model to use for data generation')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to generate')
    parser.add_argument('--topic', type=str, default=None,
                       help='Generate data for specific topic only')
    
    # Model training arguments
    parser.add_argument('--student_model', type=str, default='t5',
                       choices=['t5', 'gpt2', 'llama'],
                       help='Student model type to train')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    
    # Pipeline control
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip data generation phase')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip model training phase')
    parser.add_argument('--train_data', type=str, default=None,
                       help='Path to existing training data (if skipping generation)')
    parser.add_argument('--no_validate', action='store_true',
                       help='Skip validation steps')
    
    args = parser.parse_args()
    
    print("🚀 Medical Conversation Knowledge Distillation Pipeline")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check API keys if not skipping generation
    if not args.skip_generation:
        check_api_keys(args.teacher_model)
    
    # Phase 1: Data Generation
    train_data_path = args.train_data
    if not args.skip_generation:
        train_data_path = run_data_generation(
            args.teacher_model,
            args.num_samples,
            args.topic,
            validate=not args.no_validate
        )
    elif not train_data_path:
        print("❌ Error: Must provide --train_data when skipping generation")
        sys.exit(1)
    
    # Phase 2: Model Training
    if not args.skip_training:
        model_path = run_model_training(
            args.student_model,
            train_data_path,
            args.num_epochs,
            evaluate=not args.no_validate
        )
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Trained model saved to: {model_path}")
    else:
        print(f"\n✅ Data generation completed!")
        print(f"📁 Dataset saved to: {train_data_path}")
    
    print("\n📋 Summary:")
    print(f"  Teacher model: {args.teacher_model}")
    print(f"  Student model: {args.student_model}")
    print(f"  Samples generated: {args.num_samples}")
    print(f"  Training epochs: {args.num_epochs}")
    if args.topic:
        print(f"  Topic focus: {args.topic}")

if __name__ == "__main__":
    main() 