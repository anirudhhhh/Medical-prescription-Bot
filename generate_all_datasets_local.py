#!/usr/bin/env python3
"""
Local Medical Conversation Data Generation Script
Generates datasets for all topics using Groq teacher model
Supports GPU acceleration if available
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from data_generator import DataGenerator

def check_gpu():
    """Check if GPU is available and print info"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   Available GPUs: {gpu_count}")
        return True
    else:
        print("⚠️  No GPU detected, using CPU")
        return False

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

def check_api_key():
    """Check if GROQ API key is set"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("Please set your API key: export GROQ_API_KEY='your_api_key'")
        sys.exit(1)
    
    print("✅ GROQ API key is configured")
    return api_key

def load_config():
    """Load configuration and get topics"""
    config_path = "configs/model_configs.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    topics = config['generation_settings']['conversation_topics']
    print(f"📋 Loaded {len(topics)} topics from config")
    return topics

def generate_dataset_for_topic(generator: DataGenerator, topic: str, samples: int, output_path: str):
    """Generate dataset for a specific topic with error handling"""
    try:
        print(f"\n🔄 Generating {samples} samples for: {topic}")
        start_time = time.time()
        
        generator.generate_topic_specific_dataset(topic, samples, output_path)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Completed {topic} in {elapsed_time:.1f}s")
        
        return True
    except Exception as e:
        print(f"❌ Error generating {topic}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate medical conversation datasets for all topics')
    parser.add_argument('--samples-per-topic', type=int, default=200,
                       help='Number of samples to generate per topic (default: 200)')
    parser.add_argument('--topics', nargs='+', default=None,
                       help='Specific topics to generate (default: all topics)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip topics that already have datasets')
    parser.add_argument('--max-topics', type=int, default=None,
                       help='Maximum number of topics to process (for testing)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API calls in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    print("🚀 Medical Conversation Data Generation Pipeline (Local)")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Check API key
    api_key = check_api_key()
    
    # Load config and topics
    all_topics = load_config()
    
    # Filter topics if specified
    if args.topics:
        topics = [t for t in all_topics if t in args.topics]
        print(f"📝 Generating for {len(topics)} specified topics")
    else:
        topics = all_topics
    
    # Limit topics if specified
    if args.max_topics:
        topics = topics[:args.max_topics]
        print(f"📝 Limited to {len(topics)} topics for testing")
    
    # Initialize generator
    print(f"\n🤖 Initializing Groq teacher model...")
    generator = DataGenerator('groq')
    
    # Track progress
    successful_topics = []
    failed_topics = []
    total_samples = 0
    
    print(f"\n📊 Starting generation for {len(topics)} topics...")
    print(f"   Samples per topic: {args.samples_per_topic}")
    print(f"   Total samples to generate: {len(topics) * args.samples_per_topic}")
    
    # Generate datasets
    for i, topic in enumerate(topics, 1):
        output_path = f"data/{topic}_dataset.jsonl"
        
        # Skip if file exists and skip-existing is set
        if args.skip_existing and os.path.exists(output_path):
            print(f"⏭️  Skipping {topic} (file exists)")
            continue
        
        print(f"\n[{i}/{len(topics)}] Processing: {topic}")
        
        success = generate_dataset_for_topic(generator, topic, args.samples_per_topic, output_path)
        
        if success:
            successful_topics.append(topic)
            total_samples += args.samples_per_topic
        else:
            failed_topics.append(topic)
        
        # Add delay between topics to avoid rate limits
        if i < len(topics):
            print(f"⏳ Waiting {args.delay}s before next topic...")
            time.sleep(args.delay)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 GENERATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful topics: {len(successful_topics)}")
    print(f"❌ Failed topics: {len(failed_topics)}")
    print(f"📊 Total samples generated: {total_samples}")
    
    if successful_topics:
        print(f"\n✅ Successfully generated datasets for:")
        for topic in successful_topics:
            print(f"   - {topic}")
    
    if failed_topics:
        print(f"\n❌ Failed to generate datasets for:")
        for topic in failed_topics:
            print(f"   - {topic}")
    
    # Save summary to file
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_topics": len(topics),
        "successful_topics": successful_topics,
        "failed_topics": failed_topics,
        "total_samples": total_samples,
        "samples_per_topic": args.samples_per_topic,
        "gpu_used": gpu_available
    }
    
    summary_path = "outputs/generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary saved to: {summary_path}")
    print("\n🎉 Data generation completed!")

if __name__ == "__main__":
    main() 