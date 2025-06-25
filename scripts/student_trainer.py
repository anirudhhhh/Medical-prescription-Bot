import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

class MedicalConversationDataset(Dataset):
    def __init__(self, data_path: str, model_type: str, max_length: int = 512):
        """
        Dataset for medical conversation training
        
        Args:
            data_path: Path to JSONL dataset
            model_type: Type of model (t5, gpt2, llama)
            max_length: Maximum sequence length
        """
        self.model_type = model_type
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.model_type == 't5':
            # T5 format: input + target
            input_text = f"Extract medical information: {sample['input']}"
            target_text = sample['output']
            return {
                'input_text': input_text,
                'target_text': target_text
            }
        else:
            # GPT-2/Llama format: single text
            text = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nOutput: {sample['output']}"
            return {'text': text}

class StudentTrainer:
    def __init__(self, model_type: str, config_path: str = "configs/model_configs.json"):
        """
        Initialize student model trainer
        
        Args:
            model_type: Type of student model (t5, gpt2, llama)
            config_path: Path to configuration file
        """
        self.model_type = model_type
        self.config = self._load_config(config_path)
        self.model_config = self.config['student_models'][model_type]
        
        self.model, self.tokenizer = self._initialize_model()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _initialize_model(self):
        """Initialize model and tokenizer"""
        model_name = self.model_config['model_name']
        
        if self.model_type == 't5':
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            # Add special tokens if needed
            special_tokens = ['<medical>', '</medical>']
            tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            model.resize_token_embeddings(len(tokenizer))
            
        elif self.model_type == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # Add padding token
            tokenizer.pad_token = tokenizer.eos_token
            
        elif self.model_type == 'llama':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model, tokenizer
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for training"""
        return MedicalConversationDataset(data_path, self.model_type, self.model_config['max_length'])
    
    def train(self, train_data_path: str, output_dir: str, 
              num_epochs: int = 3, batch_size: int = None, 
              learning_rate: float = None) -> str:
        """
        Train the student model
        
        Args:
            train_data_path: Path to training data
            output_dir: Directory to save trained model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        
        Returns:
            Path to trained model
        """
        if batch_size is None:
            batch_size = self.model_config['batch_size']
        if learning_rate is None:
            learning_rate = self.model_config['learning_rate']
        
        # Prepare dataset
        dataset = self.prepare_dataset(train_data_path)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        if self.model_type == 't5':
            trainer = self._create_t5_trainer(dataset, training_args)
        else:
            trainer = self._create_causal_trainer(dataset, training_args)
        
        # Train model
        print(f"Starting training for {self.model_type} model...")
        trainer.train()
        
        # Save model
        model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        print(f"Training completed. Model saved to {model_path}")
        return model_path
    
    def _create_t5_trainer(self, dataset: Dataset, training_args: TrainingArguments):
        """Create T5 trainer"""
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples['input_text'],
                truncation=True,
                padding='max_length',
                max_length=self.model_config['max_length'],
                return_tensors='pt'
            )
            
            targets = self.tokenizer(
                examples['target_text'],
                truncation=True,
                padding='max_length',
                max_length=self.model_config['max_length'],
                return_tensors='pt'
            )
            
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': targets['input_ids']
            }
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
    
    def _create_causal_trainer(self, dataset: Dataset, training_args: TrainingArguments):
        """Create causal LM trainer (GPT-2, Llama)"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.model_config['max_length'],
                return_tensors='pt'
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
    
    def evaluate(self, test_data_path: str) -> Dict:
        """Evaluate trained model"""
        dataset = self.prepare_dataset(test_data_path)
        
        # Simple evaluation - generate outputs for a few samples
        results = {
            'total_samples': len(dataset),
            'valid_json_count': 0,
            'sample_outputs': []
        }
        
        for i in range(min(10, len(dataset))):  # Test first 10 samples
            sample = dataset[i]
            
            if self.model_type == 't5':
                input_text = sample['input_text']
                target_text = sample['target_text']
                
                # Generate output
                inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
                outputs = self.model.generate(**inputs, max_length=512)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            else:
                # For causal models, we'd need a different approach
                # This is a simplified version
                generated_text = "JSON output would be generated here"
            
            # Validate JSON
            try:
                json.loads(generated_text)
                results['valid_json_count'] += 1
            except:
                pass
            
            results['sample_outputs'].append({
                'input': sample.get('input_text', sample.get('text', '')),
                'target': sample.get('target_text', ''),
                'generated': generated_text
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Train student model using teacher-generated data')
    parser.add_argument('--model_type', type=str, default='t5', 
                       choices=['t5', 'gpt2', 'llama'],
                       help='Type of student model to train')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='models/student_model',
                       help='Output directory for trained model')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = StudentTrainer(args.model_type)
    
    # Train model
    model_path = trainer.train(
        args.train_data,
        args.output_dir,
        args.num_epochs,
        args.batch_size,
        args.learning_rate
    )
    
    # Evaluate if requested
    if args.evaluate:
        print("\nEvaluating model...")
        results = trainer.evaluate(args.train_data)  # Using train data for demo
        print(f"Evaluation Results:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Valid JSON outputs: {results['valid_json_count']}")
        print(f"  Sample outputs saved to evaluation_results.json")
        
        # Save sample outputs
        with open('evaluation_results.json', 'w') as f:
            json.dump(results['sample_outputs'], f, indent=2)

if __name__ == "__main__":
    main() 