# Medical Conversation Knowledge Distillation Pipeline

A comprehensive pipeline for generating high-quality medical conversation datasets using Groq's Llama 3 70B model and training efficient student models for medical AI applications.

## Overview

This pipeline implements knowledge distillation for medical conversation models using:
1. **Groq Teacher Model** - Uses Llama 3 70B via Groq API to generate realistic medical conversations
2. **Data Generator** - Creates comprehensive training datasets across 100+ medical topics
3. **Student Trainer** - Trains smaller, efficient models (T5, GPT-2, Llama) using teacher-generated data

## Key Features

- **100+ Medical Topics** - Comprehensive coverage from common conditions to rare diseases
- **GPU Acceleration** - Automatic GPU detection and utilization for faster training
- **Bulk Data Generation** - Generate thousands of samples across all topics efficiently
- **Quality Validation** - JSON validation and dataset statistics
- **Flexible Training** - Support for multiple student model architectures
- **Local & Cloud Support** - Run locally with GPU or use Google Colab for cloud processing

## Directory Structure

```
knowledge_distillation_pipeline/
├── configs/
│   └── model_configs.json          # Configuration for models and generation settings
├── scripts/
│   ├── teacher_model_client.py     # Groq API client for teacher model
│   ├── data_generator.py           # Dataset generation script
│   ├── student_trainer.py          # Student model training script
│   ├── metrics_evaluator.py        # Model evaluation and metrics
│   ├── run_pipeline.py             # Main pipeline orchestrator
│   └── test_pipeline.py            # Pipeline testing utilities
├── data/                           # Generated datasets (100+ topic-specific files)
├── models/                         # Trained student models
├── outputs/                        # Training outputs, metrics, and logs
├── generate_all_datasets_local.py  # Local bulk data generation script
└── requirements.txt                # Python dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd knowledge_distillation_pipeline

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY="your_groq_api_key_here"
```

### 2. Generate All Datasets (Recommended)

Generate comprehensive datasets for all 100+ medical topics:

```bash
# Generate 200 samples per topic (adjust as needed)
python generate_all_datasets_local.py --samples-per-topic 200

# For testing, generate just a few topics
python generate_all_datasets_local.py --max-topics 5 --samples-per-topic 50

# Generate specific topics only
python generate_all_datasets_local.py --topics diabetes hypertension asthma
```

### 3. Train Student Model

```bash
# Train T5 model on generated data
python scripts/student_trainer.py \
    --model_type t5 \
    --train_data data/diabetes_dataset.jsonl \
    --output_dir models/t5_student \
    --num_epochs 3 \
    --evaluate
```

## Detailed Usage

### Data Generation Options

The `generate_all_datasets_local.py` script provides flexible data generation:

```bash
# Basic usage - all topics, 200 samples each
python generate_all_datasets_local.py

# Custom number of samples per topic
python generate_all_datasets_local.py --samples-per-topic 500

# Skip topics that already have datasets
python generate_all_datasets_local.py --skip-existing

# Add delay between API calls to avoid rate limits
python generate_all_datasets_local.py --delay 2.0

# Generate only specific topics
python generate_all_datasets_local.py --topics depression anxiety insomnia

# Limit topics for testing
python generate_all_datasets_local.py --max-topics 10
```

### Individual Script Usage

Generate data for a specific topic:

```bash
python scripts/data_generator.py \
    --teacher_model groq \
    --topic hypertension \
    --num_samples 100 \
    --output_path data/hypertension_dataset.jsonl \
    --validate
```

Run the complete pipeline:

```bash
python scripts/run_pipeline.py \
    --teacher_model groq \
    --num_samples 500 \
    --student_model t5 \
    --num_epochs 3
```

### Student Model Training

Train different student model types:

```bash
# T5 model (sequence-to-sequence)
python scripts/student_trainer.py \
    --model_type t5 \
    --train_data data/diabetes_dataset.jsonl \
    --output_dir models/t5_student \
    --num_epochs 3

# GPT-2 model (causal language modeling)
python scripts/student_trainer.py \
    --model_type gpt2 \
    --train_data data/hypertension_dataset.jsonl \
    --output_dir models/gpt2_student \
    --num_epochs 5

# Llama model (causal language modeling)
python scripts/student_trainer.py \
    --model_type llama \
    --train_data data/asthma_dataset.jsonl \
    --output_dir models/llama_student \
    --num_epochs 3
```

## Supported Models

### Teacher Model
- **Groq (Llama 3 70B)** - High-performance API with fast response times

### Student Models
- **T5** - T5-base for sequence-to-sequence tasks
- **GPT-2** - GPT-2-medium for causal language modeling  
- **Llama** - Llama-2-7b for causal language modeling

## Medical Topics Covered

The pipeline generates data for 100+ medical topics including:

**Common Conditions:** hypertension, diabetes, asthma, depression, anxiety, common_cold, flu, back_pain, headaches, insomnia

**Cardiovascular:** heart_conditions, chest_pain, palpitations, stroke

**Respiratory:** pneumonia, bronchitis, covid19, tuberculosis, shortness_of_breath

**Digestive:** digestive_issues, gastroesophageal_reflux, ulcer, constipation, diarrhea, irritable_bowel_syndrome

**Neurological:** migraine, epilepsy, parkinsons_disease, multiple_sclerosis, dementia, alzheimers

**Cancer:** breast_cancer, prostate_cancer, lung_cancer, colon_cancer, skin_cancer

**Mental Health:** mental_health_crisis, suicidal_ideation, ptsd, panic_disorder, bipolar_disorder, schizophrenia

**And many more...** See `configs/model_configs.json` for the complete list.

## Data Format

Generated datasets follow this JSONL format:

```json
{
  "instruction": "Extract structured patient information (Patient Details, Medical History, Current Diagnosis, Prescription) from the following doctor-patient conversation.",
  "input": "Doctor: Hi, Mrs. Johnson. I'm Dr. Smith. How are you feeling today?\nPatient: I've been having headaches and feeling dizzy lately...",
  "output": "{\"patient_details\":{\"name\":\"Mrs. Johnson\",\"age\":55,\"gender\":\"Female\"},\"medical_history\":[\"family history of hypertension\"],\"current_diagnosis\":[\"hypertension\"],\"prescription\":[{\"medication\":\"lisinopril\",\"dosage\":\"10mg daily\",\"purpose\":\"blood pressure control\"}]}"
}
```

## Configuration

Edit `configs/model_configs.json` to customize:

- **API Settings** - Groq API configuration
- **Generation Settings** - Topics, patient moods, conversation length
- **Student Models** - Model configurations and training parameters
- **Training Settings** - Batch sizes, learning rates, etc.

## GPU Support

The pipeline automatically detects and utilizes available GPUs:

- **Automatic Detection** - Checks for CUDA-compatible GPUs
- **Fallback Support** - Uses CPU if no GPU is available
- **Memory Optimization** - Configurable batch sizes for different GPU memory capacities

## Google Colab Integration

For cloud-based processing, use the provided Colab notebook:

1. Upload `Generate_Medical_Conversations_Colab.ipynb` to Google Colab
2. Upload your project files
3. Set your Groq API key
4. Run the notebook to generate datasets in the cloud

## Output Files

### Generated Datasets
- `data/{topic}_dataset.jsonl` - Topic-specific datasets
- `data/test_dataset.jsonl` - Test dataset

### Training Outputs
- `models/{model_type}_student/` - Trained model checkpoints
- `outputs/{model_type}_sample_outputs.json` - Sample model outputs
- `outputs/generation_summary.json` - Data generation summary
- `outputs/comprehensive_metrics.json` - Model evaluation metrics

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Use `--delay` parameter to add delays between API calls
   - Monitor your Groq API usage

2. **Memory Issues**
   - Reduce batch size in student model configuration
   - Use smaller models for limited GPU memory

3. **Import Errors**
   - Ensure you're running from the project root directory
   - Check that all dependencies are installed

4. **JSON Validation Errors**
   - Check teacher model prompts in `teacher_model_client.py`
   - Verify API responses are properly formatted

### Performance Tips

- **GPU Usage**: Ensure CUDA is properly installed for GPU acceleration
- **Batch Processing**: Use `--skip-existing` to resume interrupted generation
- **API Efficiency**: Adjust `--delay` based on your Groq API tier limits

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

### Batch Processing
Generate datasets in batches to manage API costs:

```bash
# Generate first 20 topics
python generate_all_datasets_local.py --max-topics 20

# Generate next 20 topics (skip existing)
python generate_all_datasets_local.py --max-topics 40 --skip-existing
```

## API Costs

- **Groq API**: Pay-per-token pricing, typically $0.05-0.10 per 1M tokens
- **Estimated Cost**: ~$10-50 for generating 20,000 samples (200 per topic × 100 topics)
- **Cost Control**: Use `--max-topics` and `--samples-per-topic` to control generation volume

## Next Steps

- **Model Evaluation**: Use `metrics_evaluator.py` for comprehensive model assessment
- **Fine-tuning**: Experiment with different student model architectures
- **Deployment**: Export trained models for production use
- **Scaling**: Use cloud platforms for larger-scale training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 