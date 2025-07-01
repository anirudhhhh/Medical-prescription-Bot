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

## Step-by-Step Usage Guide

### 1. Prepare for Data Generation

- **Ensure the `data/` folder is empty** before starting a new generation run. You can clear it with:
  ```sh
  rm -rf data/*
  ```
- **Obtain a Groq API key** from your Groq account.
- **Set the API key when prompted** (or export it in your terminal):
  ```sh
  export GROQ_API_KEY="your_groq_api_key_here"
  ```

### 2. Generate Training Data

- Run the data generation script. You will be prompted for your Groq API key if it is not set:
  ```sh
  python3 generate_all_datasets_local.py --samples-per-topic 200 --resume
  ```
- **If your Groq API key runs out or expires during generation:**
  1. Obtain a new Groq API key.
  2. Set the new key:
     ```sh
     export GROQ_API_KEY="your_new_groq_api_key"
     ```
  3. Rerun the data generator with the same command and the `--resume` flag:
     ```sh
     python3 generate_all_datasets_local.py --samples-per-topic 200 --resume
     ```
  - The script will skip completed topics and regenerate any partial datasets, ensuring data integrity.

### 3. Train the Student Model

- Once data generation is complete, train a student model (e.g., T5) using the generated data:
  ```sh
  python3 scripts/student_trainer.py \
      --model_type t5 \
      --train_data data/diabetes_dataset.jsonl \
      --output_dir models/t5_diabetes_student \
      --num_epochs 3 \
      --evaluate
  ```
- Replace `data/diabetes_dataset.jsonl` with the dataset for your topic of interest.

### 4. Test/Evaluate the Model

- After training, you can evaluate the model using the metrics evaluator script:
  ```sh
  python3 scripts/metrics_evaluator.py
  ```
- Review the output files in the `outputs/` directory for evaluation results and reports.

---

## Troubleshooting & Tips

- **Always start with an empty `data/` folder for a new generation run.**
- **Use the `--resume` flag** to safely continue interrupted data generation.
- **Set your Groq API key** before running any script that requires it.
- **If you encounter an authentication error,** get a new API key and rerun the generator with `--resume`.
- **Monitor the `outputs/` folder** for generation summaries and evaluation reports.

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