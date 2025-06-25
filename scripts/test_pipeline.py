#!/usr/bin/env python3
"""
Test script for the knowledge distillation pipeline
Generates mock data and tests pipeline components
"""

import json
import os
import random
from typing import Dict, List
from pathlib import Path

def generate_mock_conversation(topic: str, mood: str, min_exchanges: int = 3, max_exchanges: int = 8) -> str:
    """Generate a mock medical conversation with variable length"""
    patient_names = ["John Smith", "Mary Johnson", "David Wilson", "Sarah Brown", "Michael Davis"]
    patient_name = random.choice(patient_names)
    age = random.randint(25, 75)

    # Pools of utterances for each topic
    utterances = {
        "hypertension": {
            "doctor": [
                f"Hello {patient_name}, how are you feeling today?",
                "How long have you been experiencing these symptoms?",
                "Let's check your blood pressure today and discuss some lifestyle changes.",
                "What readings have you been getting at home?",
                "That is elevated. Let's monitor this and consider medication if needed.",
                "Are you taking your medication regularly?",
                "Do you have any headaches or dizziness?",
                "Let's talk about reducing salt in your diet."
            ],
            "patient": [
                "I've been having headaches and feeling dizzy lately.",
                "About two weeks now. I'm worried because my father had heart problems.",
                "I'm concerned about my blood pressure readings at home.",
                "Usually around 150/95. I'm trying to reduce salt but it's hard.",
                "Yes, but sometimes I forget.",
                "I feel tired and sometimes get blurry vision.",
                "I try to walk every day but it's difficult.",
                "I don't like taking too many pills."
            ]
        },
        "diabetes": {
            "doctor": [
                f"Hi {patient_name}, how's your diabetes management going?",
                "What are your typical morning blood sugar readings?",
                "Let's adjust your medication timing and review your meal schedule.",
                "Have you noticed any changes in your vision or foot numbness?",
                "Let's check your A1C and do a comprehensive eye exam.",
                "Are you following the diet plan?",
                "How often are you exercising?",
                "Do you have any episodes of low blood sugar?"
            ],
            "patient": [
                "I've been checking my blood sugar regularly, but it's still high in the mornings.",
                "Usually between 180-220. I'm following the diet plan.",
                "Yes, I've been feeling more tired than usual and drinking a lot of water.",
                "My vision seems a bit blurry sometimes.",
                "I try to walk after dinner.",
                "Sometimes I feel shaky before lunch.",
                "I find it hard to avoid sweets.",
                "I haven't had any foot numbness."
            ]
        },
        "asthma": {
            "doctor": [
                f"Good afternoon {patient_name}, how's your breathing been?",
                "How many times a day are you using your inhaler?",
                "That's more than we want. Let's adjust your medication and check your peak flow.",
                "Do you hear any wheezing?",
                "Let's do a spirometry test and review your asthma action plan.",
                "Are you having symptoms at night?",
                "Do you have any allergies?",
                "Have you missed any doses of your controller inhaler?"
            ],
            "patient": [
                "I've been using my inhaler more often, especially at night.",
                "About 4-5 times, and I wake up coughing.",
                "Yes, especially when I exercise or when it's cold outside.",
                "Sometimes, and I feel tightness in my chest.",
                "I have a cat at home and sometimes sneeze a lot.",
                "I forgot to take my inhaler yesterday.",
                "I feel better after using the rescue inhaler.",
                "I get short of breath when climbing stairs."
            ]
        }
    }
    # Default to hypertension if topic not found
    topic_utter = utterances.get(topic, utterances["hypertension"])
    num_exchanges = random.randint(min_exchanges, max_exchanges)
    conversation = []
    for i in range(num_exchanges):
        if i % 2 == 0:
            # Doctor's turn
            utter = random.choice(topic_utter["doctor"])
            conversation.append(f"Doctor: {utter}")
        else:
            # Patient's turn
            utter = random.choice(topic_utter["patient"])
            conversation.append(f"Patient: {utter}")
    return "\n".join(conversation)

def generate_mock_structured_output(conversation: str, topic: str) -> str:
    """Generate mock structured JSON output"""
    patient_names = ["John Smith", "Mary Johnson", "David Wilson", "Sarah Brown", "Michael Davis"]
    patient_name = random.choice(patient_names)
    age = random.randint(25, 75)
    gender = random.choice(["Male", "Female"])
    
    outputs = {
        "hypertension": {
            "patient_details": {
                "name": patient_name,
                "age": age,
                "gender": gender
            },
            "medical_history": [
                "Family history of heart disease",
                "Previous diagnosis of hypertension"
            ],
            "current_diagnosis": [
                "Essential hypertension",
                "Elevated blood pressure readings"
            ],
            "prescription": [
                {
                    "medication": "Lisinopril",
                    "dosage": "10mg daily",
                    "purpose": "Blood pressure control",
                    "instructions": "Take in the morning"
                },
                {
                    "medication": "Lifestyle modifications",
                    "dosage": "N/A",
                    "purpose": "Reduce blood pressure",
                    "instructions": "Reduce salt intake, exercise regularly"
                }
            ]
        },
        "diabetes": {
            "patient_details": {
                "name": patient_name,
                "age": age,
                "gender": gender
            },
            "medical_history": [
                "Type 2 diabetes",
                "Previous blood sugar monitoring"
            ],
            "current_diagnosis": [
                "Poorly controlled diabetes",
                "Elevated morning blood glucose"
            ],
            "prescription": [
                {
                    "medication": "Metformin",
                    "dosage": "500mg twice daily",
                    "purpose": "Blood glucose control",
                    "instructions": "Take with meals"
                },
                {
                    "medication": "Blood glucose monitoring",
                    "dosage": "N/A",
                    "purpose": "Track blood sugar levels",
                    "instructions": "Check fasting and post-meal levels"
                }
            ]
        },
        "asthma": {
            "patient_details": {
                "name": patient_name,
                "age": age,
                "gender": gender
            },
            "medical_history": [
                "Asthma diagnosis",
                "Previous inhaler use"
            ],
            "current_diagnosis": [
                "Poorly controlled asthma",
                "Increased rescue inhaler use"
            ],
            "prescription": [
                {
                    "medication": "Albuterol inhaler",
                    "dosage": "2 puffs as needed",
                    "purpose": "Quick relief of asthma symptoms",
                    "instructions": "Use before exercise and when symptoms occur"
                },
                {
                    "medication": "Fluticasone inhaler",
                    "dosage": "2 puffs twice daily",
                    "purpose": "Long-term asthma control",
                    "instructions": "Use regularly to prevent symptoms"
                }
            ]
        }
    }
    
    return json.dumps(outputs.get(topic, outputs["hypertension"]))

def generate_mock_dataset(num_samples: int = 50, topics: List[str] = None, min_exchanges: int = 3, max_exchanges: int = 8) -> List[Dict]:
    """Generate a mock dataset for testing"""
    if topics is None:
        topics = ["hypertension", "diabetes", "asthma"]
    
    moods = ["worried", "calm", "panicked", "relaxed", "concerned", "optimistic"]
    samples = []
    
    for i in range(num_samples):
        topic = random.choice(topics)
        mood = random.choice(moods)
        
        conversation = generate_mock_conversation(topic, mood, min_exchanges, max_exchanges)
        structured_output = generate_mock_structured_output(conversation, topic)
        
        sample = {
            "instruction": "Extract structured patient information (Patient Details, Medical History, Current Diagnosis, Prescription) from the following doctor-patient conversation.",
            "input": conversation,
            "output": structured_output
        }
        samples.append(sample)
    
    return samples

def save_dataset(samples: List[Dict], output_path: str):
    """Save dataset to JSONL file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(samples)} samples to {output_path}")

def calculate_metrics(samples: List[Dict]) -> Dict:
    """Calculate comprehensive metrics for the dataset"""
    metrics = {
        "total_samples": len(samples),
        "topics_covered": {},
        "conversation_lengths": [],
        "json_validity": 0,
        "medical_terms": {},
        "patient_demographics": {"ages": [], "genders": []},
        "prescription_stats": {"total_prescriptions": 0, "avg_prescriptions_per_sample": 0},
        "quality_scores": {"completeness": 0, "consistency": 0}
    }
    
    total_prescriptions = 0
    valid_json_count = 0
    completeness_scores = []
    
    for sample in samples:
        # Check JSON validity
        try:
            json.loads(sample['output'])
            valid_json_count += 1
        except:
            pass
        
        # Analyze conversation length
        conversation = sample['input']
        exchanges = conversation.count('Doctor:') + conversation.count('Patient:')
        metrics['conversation_lengths'].append(exchanges)
        
        # Extract topic from conversation
        conversation_lower = conversation.lower()
        if any(term in conversation_lower for term in ['blood pressure', 'hypertension', 'pressure']):
            metrics['topics_covered']['hypertension'] = metrics['topics_covered'].get('hypertension', 0) + 1
        if any(term in conversation_lower for term in ['diabetes', 'blood sugar', 'glucose']):
            metrics['topics_covered']['diabetes'] = metrics['topics_covered'].get('diabetes', 0) + 1
        if any(term in conversation_lower for term in ['asthma', 'breathing', 'inhaler']):
            metrics['topics_covered']['asthma'] = metrics['topics_covered'].get('asthma', 0) + 1
        
        # Count medical terms
        medical_terms = ['symptoms', 'diagnosis', 'treatment', 'medication', 'prescription', 'blood pressure', 'diabetes', 'asthma']
        for term in medical_terms:
            if term in conversation_lower:
                metrics['medical_terms'][term] = metrics['medical_terms'].get(term, 0) + 1
        
        # Analyze structured output
        try:
            output_data = json.loads(sample['output'])
            
            # Patient demographics
            if 'patient_details' in output_data:
                details = output_data['patient_details']
                if 'age' in details and isinstance(details['age'], int):
                    metrics['patient_demographics']['ages'].append(details['age'])
                if 'gender' in details:
                    metrics['patient_demographics']['genders'].append(details['gender'])
            
            # Prescription analysis
            if 'prescription' in output_data:
                prescriptions = output_data['prescription']
                total_prescriptions += len(prescriptions)
                
                # Completeness score
                required_fields = ['patient_details', 'medical_history', 'current_diagnosis', 'prescription']
                completeness = sum(1 for field in required_fields if field in output_data) / len(required_fields)
                completeness_scores.append(completeness)
                
        except:
            pass
    
    # Calculate averages and final metrics
    metrics['json_validity'] = valid_json_count / len(samples) if samples else 0
    metrics['avg_conversation_length'] = sum(metrics['conversation_lengths']) / len(metrics['conversation_lengths']) if metrics['conversation_lengths'] else 0
    metrics['prescription_stats']['total_prescriptions'] = total_prescriptions
    metrics['prescription_stats']['avg_prescriptions_per_sample'] = total_prescriptions / len(samples) if samples else 0
    metrics['quality_scores']['completeness'] = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
    
    # Consistency score (how many samples have similar structure)
    if completeness_scores:
        consistency = 1 - (max(completeness_scores) - min(completeness_scores))
        metrics['quality_scores']['consistency'] = max(0, consistency)
    
    return metrics

def print_metrics(metrics: Dict):
    """Print formatted metrics"""
    print("\n" + "="*60)
    print("📊 DATASET METRICS REPORT")
    print("="*60)
    
    print(f"\n📈 Basic Statistics:")
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  JSON validity rate: {metrics['json_validity']:.2%}")
    print(f"  Average conversation length: {metrics['avg_conversation_length']:.1f} exchanges")
    
    print(f"\n🏥 Topic Distribution:")
    for topic, count in metrics['topics_covered'].items():
        percentage = (count / metrics['total_samples']) * 100
        print(f"  {topic.capitalize()}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n💊 Prescription Analysis:")
    print(f"  Total prescriptions: {metrics['prescription_stats']['total_prescriptions']}")
    print(f"  Average prescriptions per sample: {metrics['prescription_stats']['avg_prescriptions_per_sample']:.1f}")
    
    print(f"\n👥 Patient Demographics:")
    if metrics['patient_demographics']['ages']:
        avg_age = sum(metrics['patient_demographics']['ages']) / len(metrics['patient_demographics']['ages'])
        print(f"  Average age: {avg_age:.1f} years")
    if metrics['patient_demographics']['genders']:
        gender_counts = {}
        for gender in metrics['patient_demographics']['genders']:
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count} patients")
    
    print(f"\n🔍 Medical Terms Frequency:")
    sorted_terms = sorted(metrics['medical_terms'].items(), key=lambda x: x[1], reverse=True)
    for term, count in sorted_terms[:10]:  # Top 10 terms
        percentage = (count / metrics['total_samples']) * 100
        print(f"  {term}: {count} occurrences ({percentage:.1f}%)")
    
    print(f"\n⭐ Quality Scores:")
    print(f"  Completeness: {metrics['quality_scores']['completeness']:.2%}")
    print(f"  Consistency: {metrics['quality_scores']['consistency']:.2%}")
    
    print("\n" + "="*60)

def main():
    """Main test function"""
    print("🧪 Testing Knowledge Distillation Pipeline")
    print("="*60)
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    # Generate test dataset
    print("\n📝 Generating mock dataset...")
    test_samples = generate_mock_dataset(num_samples=50, topics=["hypertension", "diabetes", "asthma"])
    
    # Save dataset
    output_path = "data/test_dataset.jsonl"
    save_dataset(test_samples, output_path)
    
    # Calculate and display metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(test_samples)
    print_metrics(metrics)
    
    # Save metrics to file
    metrics_path = "outputs/test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to {metrics_path}")
    
    # Show sample conversation
    print(f"\n📋 Sample Conversation:")
    print("-" * 40)
    sample = test_samples[0]
    print(f"Topic: {list(metrics['topics_covered'].keys())[0]}")
    print(f"Conversation:\n{sample['input']}")
    print(f"\nStructured Output:\n{sample['output']}")
    
    print(f"\n✅ Test completed successfully!")
    print(f"📁 Generated {len(test_samples)} test samples")
    print(f"📊 Comprehensive metrics calculated")

if __name__ == "__main__":
    main() 