import json
import os
import time
import random
from typing import List, Dict
from tqdm import tqdm
import argparse
from teacher_model_client import TeacherModelClient

class DataGenerator:
    def __init__(self, teacher_model: str, config_path: str = "configs/model_configs.json"):
        """
        Initialize data generator
        
        Args:
            teacher_model: Name of the teacher model to use
            config_path: Path to configuration file
        """
        self.teacher_client = TeacherModelClient(teacher_model, config_path)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def generate_dataset(self, output_path: str, num_samples: int = None) -> str:
        """
        Generate complete training dataset
        
        Args:
            output_path: Path to save the generated dataset
            num_samples: Number of samples to generate (default from config)
        
        Returns:
            Path to generated dataset
        """
        if num_samples is None:
            num_samples = self.config['generation_settings']['total_samples']
        
        print(f"Generating {num_samples} training samples using {self.teacher_client.model_name}...")
        
        samples = []
        topics = self.config['generation_settings']['conversation_topics']
        moods = self.config['generation_settings']['patient_moods']
        length_range = self.config['generation_settings']['conversation_length']
        
        # Calculate samples per topic
        samples_per_topic = num_samples // len(topics)
        remaining_samples = num_samples % len(topics)
        
        with tqdm(total=num_samples, desc="Generating samples") as pbar:
            for i, topic in enumerate(topics):
                topic_samples = samples_per_topic + (1 if i < remaining_samples else 0)
                
                for _ in range(topic_samples):
                    # Randomly select mood
                    mood = random.choice(moods)
                    
                    # Generate sample
                    sample = self.teacher_client.generate_complete_sample(topic, mood, length_range)
                    
                    if sample:
                        samples.append(sample)
                        pbar.update(1)
                    
                    # Add delay to avoid rate limits
                    time.sleep(1)
        
        # Save dataset
        self._save_dataset(samples, output_path)
        
        print(f"Generated {len(samples)} samples. Saved to {output_path}")
        return output_path
    
    def generate_topic_specific_dataset(self, topic: str, num_samples: int, output_path: str) -> str:
        """
        Generate dataset for a specific medical topic
        
        Args:
            topic: Medical topic to focus on
            num_samples: Number of samples to generate
            output_path: Path to save the dataset
        
        Returns:
            Path to generated dataset
        """
        print(f"Generating {num_samples} samples for topic: {topic}")
        
        samples = []
        moods = self.config['generation_settings']['patient_moods']
        length_range = self.config['generation_settings']['conversation_length']
        
        with tqdm(total=num_samples, desc=f"Generating {topic} samples") as pbar:
            for _ in range(num_samples):
                mood = random.choice(moods)
                sample = self.teacher_client.generate_complete_sample(topic, mood, length_range)
                
                if sample:
                    samples.append(sample)
                    pbar.update(1)
                
                time.sleep(1)
        
        self._save_dataset(samples, output_path)
        print(f"Generated {len(samples)} samples for {topic}. Saved to {output_path}")
        return output_path
    
    def _save_dataset(self, samples: List[Dict], output_path: str):
        """Save dataset to JSONL file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
    
    def validate_dataset(self, dataset_path: str) -> Dict:
        """
        Validate generated dataset
        
        Returns:
            Dictionary with validation statistics
        """
        samples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        
        # Validation statistics
        stats = {
            'total_samples': len(samples),
            'valid_json_count': 0,
            'topics_covered': set(),
            'avg_conversation_length': 0,
            'errors': []
        }
        
        total_length = 0
        
        for i, sample in enumerate(samples):
            try:
                # Validate JSON output
                json.loads(sample['output'])
                stats['valid_json_count'] += 1
                
                # Count conversation exchanges
                conversation = sample['input']
                exchanges = conversation.count('Doctor:') + conversation.count('Patient:')
                total_length += exchanges
                
                # Extract topic from conversation (simple heuristic)
                if any(word in conversation.lower() for word in ['blood pressure', 'hypertension']):
                    stats['topics_covered'].add('hypertension')
                elif any(word in conversation.lower() for word in ['diabetes', 'blood sugar']):
                    stats['topics_covered'].add('diabetes')
                # Add more topic detection as needed
                
            except Exception as e:
                stats['errors'].append(f"Sample {i}: {str(e)}")
        
        stats['avg_conversation_length'] = total_length / len(samples) if samples else 0
        stats['topics_covered'] = list(stats['topics_covered'])
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Generate medical conversation dataset using teacher model')
    parser.add_argument('--teacher_model', type=str, default='deepseek', 
                       choices=['deepseek', 'groq', 'openai', 'anthropic'],
                       help='Teacher model to use')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to generate')
    parser.add_argument('--topic', type=str, default=None,
                       help='Generate samples for specific topic only')
    parser.add_argument('--output_path', type=str, default='data/generated_dataset.jsonl',
                       help='Output path for generated dataset')
    parser.add_argument('--validate', action='store_true',
                       help='Validate generated dataset')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DataGenerator(args.teacher_model)
    
    # Generate dataset
    if args.topic:
        generator.generate_topic_specific_dataset(args.topic, args.num_samples or 100, args.output_path)
    else:
        generator.generate_dataset(args.output_path, args.num_samples)
    
    # Validate if requested
    if args.validate:
        print("\nValidating dataset...")
        stats = generator.validate_dataset(args.output_path)
        print(f"Validation Results:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Valid JSON: {stats['valid_json_count']}")
        print(f"  Average conversation length: {stats['avg_conversation_length']:.1f} exchanges")
        print(f"  Topics covered: {stats['topics_covered']}")
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")

if __name__ == "__main__":
    main() 