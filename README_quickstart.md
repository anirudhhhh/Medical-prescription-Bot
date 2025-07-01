# Quick Start: Knowledge Distillation Pipeline

## 1. Set Up Your Environment

(Optional but recommended) Create and activate a virtual environment:
```sh
python3 -m venv venv
source venv/bin/activate
```

Install all required dependencies:
```sh
pip install -r requirements.txt
```

## 2. Prepare Data Directory
```sh
rm -rf data/*
```

## 3. Set Groq API Key
```sh
export GROQ_API_KEY="your_groq_api_key_here"
```

## 4. Generate Training Data
```sh
python3 generate_all_datasets_local.py --samples-per-topic 200 --resume
```
If your API key expires, get a new one, set it, and rerun the above command with `--resume`.

## 5. Train the Student Model (Single Topic Example)
- The default student model is now `t5-small` for lower memory usage.
- You can set the batch size with `--batch_size` (e.g., 4).
```sh
python3 scripts/student_trainer.py \
    --model_type t5 \
    --train_data data/diabetes_dataset.jsonl \
    --output_dir models/t5_diabetes_student \
    --num_epochs 3 \
    --batch_size 4 \
    --evaluate
```

## 6. Train a Student Model on All Topics Combined
```sh
cat data/*_dataset.jsonl > data/all_topics_dataset.jsonl
python3 scripts/student_trainer.py \
    --model_type t5 \
    --train_data data/all_topics_dataset.jsonl \
    --output_dir models/t5_all_topics_student \
    --num_epochs 3 \
    --batch_size 4 \
    --evaluate
```

## 7. Evaluate the Model
```sh
python3 scripts/metrics_evaluator.py
```

## 8. Save/Export the Model
- The trained model and tokenizer are saved automatically to the specified `--output_dir` (e.g., `models/t5_all_topics_student/final_model`).
- You can load this directory later for inference or further fine-tuning. 