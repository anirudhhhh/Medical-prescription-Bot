# Medical Conversation Knowledge Distillation Pipeline

This pipeline implements knowledge distillation for training medical conversation models using teacher models (DeepSeek, Groq, OpenAI, Anthropic) to generate high-quality training data.

## Overview

The pipeline consists of three main components:
1. **Teacher Model Client** - Interfaces with various teacher models to generate conversations and structured outputs
2. **Data Generator** - Creates complete training datasets using teacher models
3. **Student Trainer** - Trains smaller student models using the teacher-generated data

## Directory Structure

```
knowledge_distillation_pipeline/
├── configs/
│   └── model_configs.json          # Configuration for teacher and student models
├── scripts/
│   ├── teacher_model_client.py     # Teacher model API client
│   ├── data_generator.py           # Dataset generation script
│   └── student_trainer.py          # Student model training script
├── data/                           # Generated datasets
├── models/                         # Trained student models
├── outputs/                        # Training outputs and logs
└── requirements.txt                # Python dependencies
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   export DEEPSEEK_API_KEY="your_deepseek_key"
   export GROQ_API_KEY="your_groq_key"
   export OPENAI_API_KEY="your_openai_key"
   export ANTHROPIC_API_KEY="your_anthropic_key"
   ```

## Usage

### 1. Generate Training Data

Generate a complete dataset using a teacher model:

```bash
cd scripts
python data_generator.py \
    --teacher_model deepseek \
    --num_samples 1000 \
    --output_path ../data/generated_dataset.jsonl \
    --validate
```

Generate data for a specific topic:

```bash
python data_generator.py \
    --teacher_model groq \
    --topic hypertension \
    --num_samples 100 \
    --output_path ../data/hypertension_dataset.jsonl
```

### 2. Train Student Model

Train a T5 student model:

```bash
python student_trainer.py \
    --model_type t5 \
    --train_data ../data/generated_dataset.jsonl \
    --output_dir ../models/t5_student \
    --num_epochs 3 \
    --evaluate
```

Train a GPT-2 student model:

```bash
python student_trainer.py \
    --model_type gpt2 \
    --train_data ../data/generated_dataset.jsonl \
    --output_dir ../models/gpt2_student \
    --num_epochs 5
```

### 3. Complete Pipeline Example

```bash
# Step 1: Generate dataset
python data_generator.py --teacher_model deepseek --num_samples 500 --validate

# Step 2: Train student model
python student_trainer.py --model_type t5 --train_data ../data/generated_dataset.jsonl --evaluate
```

## Supported Models

### Teacher Models
- **DeepSeek** - DeepSeek Chat API
- **Groq** - Groq API (Llama 3 70B)
- **OpenAI** - GPT-4 API
- **Anthropic** - Claude API

### Student Models
- **T5** - T5-base for sequence-to-sequence tasks
- **GPT-2** - GPT-2-medium for causal language modeling
- **Llama** - Llama-2-7b for causal language modeling

## Configuration

Edit `configs/model_configs.json` to customize:
- API endpoints and parameters
- Model configurations
- Generation settings (topics, moods, conversation length)
- Training parameters

## Data Format

The generated dataset follows this JSONL format:

```json
{
  "instruction": "Extract structured patient information...",
  "input": "Doctor: Hello, how are you feeling today?\nPatient: I've been having headaches...",
  "output": "{\"patient_details\":{\"name\":\"John Doe\",\"age\":45,\"gender\":\"Male\"},...}"
}
```

## Features

- **Multi-model support** - Easy switching between different teacher and student models
- **Quality validation** - JSON validation and dataset statistics
- **Flexible generation** - Topic-specific or general datasets
- **Robust training** - Support for different model architectures
- **Error handling** - Retry logic and graceful failure handling

## Advanced Usage

### Custom Topics
Add new medical topics in `configs/model_configs.json`:

```json
"conversation_topics": [
  "hypertension", "diabetes", "asthma", "your_custom_topic"
]
```

### Custom Student Models
Add new student model configurations:

```json
"student_models": {
  "your_model": {
    "model_name": "your/model/path",
    "max_length": 512,
    "batch_size": 8,
    "learning_rate": 5e-5
  }
}
```

## Troubleshooting

1. **API Rate Limits**: The pipeline includes delays between API calls. Adjust in `teacher_model_client.py`
2. **Memory Issues**: Reduce batch size for student models in configuration
3. **JSON Validation Errors**: Check teacher model prompts in `teacher_model_client.py`

## Next Steps

- Add RLHF (Reinforcement Learning from Human Feedback)
- Implement preference modeling
- Add more sophisticated evaluation metrics
- Support for more model architectures 