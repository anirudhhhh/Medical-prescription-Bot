#!/usr/bin/env python3
"""
Comprehensive metrics evaluator for the knowledge distillation pipeline
Provides detailed analysis of generated datasets and model performance
"""

import json
import re
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MetricsEvaluator:
    def __init__(self):
        """Initialize the metrics evaluator"""
        self.metrics = {}
        
    def evaluate_dataset(self, samples: List[Dict]) -> Dict:
        """
        Comprehensive dataset evaluation
        
        Args:
            samples: List of training samples
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            "basic_stats": self._calculate_basic_stats(samples),
            "conversation_analysis": self._analyze_conversations(samples),
            "structured_output_analysis": self._analyze_structured_outputs(samples),
            "medical_content_analysis": self._analyze_medical_content(samples),
            "quality_metrics": self._calculate_quality_metrics(samples),
            "diversity_metrics": self._calculate_diversity_metrics(samples),
            "consistency_metrics": self._calculate_consistency_metrics(samples)
        }
        
        self.metrics = metrics
        return metrics
    
    def _calculate_basic_stats(self, samples: List[Dict]) -> Dict:
        """Calculate basic statistics"""
        total_samples = len(samples)
        
        # JSON validity
        valid_json_count = 0
        for sample in samples:
            try:
                json.loads(sample['output'])
                valid_json_count += 1
            except:
                pass
        
        # Conversation lengths
        conversation_lengths = []
        for sample in samples:
            conversation = sample['input']
            exchanges = conversation.count('Doctor:') + conversation.count('Patient:')
            conversation_lengths.append(exchanges)
        
        return {
            "total_samples": total_samples,
            "json_validity_rate": valid_json_count / total_samples if total_samples > 0 else 0,
            "avg_conversation_length": np.mean(conversation_lengths) if conversation_lengths else 0,
            "min_conversation_length": min(conversation_lengths) if conversation_lengths else 0,
            "max_conversation_length": max(conversation_lengths) if conversation_lengths else 0,
            "std_conversation_length": np.std(conversation_lengths) if conversation_lengths else 0
        }
    
    def _analyze_conversations(self, samples: List[Dict]) -> Dict:
        """Analyze conversation patterns"""
        doctor_turns = []
        patient_turns = []
        topics_detected = defaultdict(int)
        medical_terms = Counter()
        
        # Medical topics and terms
        topic_keywords = {
            "hypertension": ["blood pressure", "hypertension", "pressure", "heart"],
            "diabetes": ["diabetes", "blood sugar", "glucose", "insulin"],
            "asthma": ["asthma", "breathing", "inhaler", "wheezing"],
            "depression": ["depression", "mood", "sad", "anxiety"],
            "pain": ["pain", "ache", "hurt", "discomfort"],
            "fever": ["fever", "temperature", "hot", "chills"]
        }
        
        for sample in samples:
            conversation = sample['input'].lower()
            
            # Count turns
            doctor_count = conversation.count('doctor:')
            patient_count = conversation.count('patient:')
            doctor_turns.append(doctor_count)
            patient_turns.append(patient_count)
            
            # Detect topics
            for topic, keywords in topic_keywords.items():
                if any(keyword in conversation for keyword in keywords):
                    topics_detected[topic] += 1
            
            # Count medical terms
            medical_terms_list = [
                "symptoms", "diagnosis", "treatment", "medication", "prescription",
                "blood pressure", "diabetes", "asthma", "pain", "fever", "headache",
                "dizziness", "nausea", "fatigue", "cough", "shortness of breath"
            ]
            
            for term in medical_terms_list:
                if term in conversation:
                    medical_terms[term] += 1
        
        return {
            "avg_doctor_turns": np.mean(doctor_turns) if doctor_turns else 0,
            "avg_patient_turns": np.mean(patient_turns) if patient_turns else 0,
            "topics_detected": dict(topics_detected),
            "medical_terms_frequency": dict(medical_terms.most_common(20)),
            "conversation_balance": np.mean(patient_turns) / np.mean(doctor_turns) if doctor_turns and np.mean(doctor_turns) > 0 else 0
        }
    
    def _analyze_structured_outputs(self, samples: List[Dict]) -> Dict:
        """Analyze structured JSON outputs"""
        field_completeness = defaultdict(int)
        prescription_stats = []
        patient_demographics = {"ages": [], "genders": []}
        diagnosis_variety = Counter()
        
        for sample in samples:
            try:
                output_data = json.loads(sample['output'])
                
                # Field completeness
                required_fields = ['patient_details', 'medical_history', 'current_diagnosis', 'prescription']
                for field in required_fields:
                    if field in output_data:
                        field_completeness[field] += 1
                
                # Prescription analysis
                if 'prescription' in output_data:
                    prescriptions = output_data['prescription']
                    prescription_stats.append(len(prescriptions))
                    
                    # Analyze prescription structure
                    for prescription in prescriptions:
                        if isinstance(prescription, dict):
                            for key in ['medication', 'dosage', 'purpose', 'instructions']:
                                if key in prescription:
                                    field_completeness[f"prescription_{key}"] += 1
                
                # Patient demographics
                if 'patient_details' in output_data:
                    details = output_data['patient_details']
                    if 'age' in details and isinstance(details['age'], (int, float)):
                        patient_demographics['ages'].append(details['age'])
                    if 'gender' in details:
                        patient_demographics['genders'].append(details['gender'])
                
                # Diagnosis variety
                if 'current_diagnosis' in output_data:
                    diagnoses = output_data['current_diagnosis']
                    if isinstance(diagnoses, list):
                        for diagnosis in diagnoses:
                            diagnosis_variety[diagnosis] += 1
                            
            except:
                continue
        
        total_samples = len(samples)
        
        return {
            "field_completeness": {field: count/total_samples for field, count in field_completeness.items()},
            "avg_prescriptions_per_sample": np.mean(prescription_stats) if prescription_stats else 0,
            "patient_demographics": {
                "avg_age": np.mean(patient_demographics['ages']) if patient_demographics['ages'] else 0,
                "gender_distribution": Counter(patient_demographics['genders']),
                "age_range": {
                    "min": min(patient_demographics['ages']) if patient_demographics['ages'] else 0,
                    "max": max(patient_demographics['ages']) if patient_demographics['ages'] else 0
                }
            },
            "diagnosis_variety": dict(diagnosis_variety.most_common(10))
        }
    
    def _analyze_medical_content(self, samples: List[Dict]) -> Dict:
        """Analyze medical content quality"""
        medical_accuracy_indicators = {
            "proper_medication_names": 0,
            "realistic_dosages": 0,
            "appropriate_diagnoses": 0,
            "logical_symptom_diagnosis_pairs": 0
        }
        
        # Common medical terms and realistic values
        common_medications = [
            "metformin", "lisinopril", "atorvastatin", "amlodipine", "omeprazole",
            "albuterol", "fluticasone", "ibuprofen", "acetaminophen", "aspirin"
        ]
        
        realistic_dosages = [
            "10mg", "20mg", "50mg", "100mg", "500mg", "1000mg",
            "once daily", "twice daily", "as needed", "every 4-6 hours"
        ]
        
        for sample in samples:
            conversation = sample['input'].lower()
            try:
                output_data = json.loads(sample['output'])
                
                # Check for proper medication names
                if 'prescription' in output_data:
                    for prescription in output_data['prescription']:
                        if isinstance(prescription, dict) and 'medication' in prescription:
                            med_name = prescription['medication'].lower()
                            if any(common_med in med_name for common_med in common_medications):
                                medical_accuracy_indicators["proper_medication_names"] += 1
                            
                            if 'dosage' in prescription:
                                dosage = prescription['dosage'].lower()
                                if any(realistic_dose in dosage for realistic_dose in realistic_dosages):
                                    medical_accuracy_indicators["realistic_dosages"] += 1
                
                # Check for logical symptom-diagnosis pairs
                if any(symptom in conversation for symptom in ["headache", "dizziness"]) and \
                   any(diagnosis in str(output_data).lower() for diagnosis in ["hypertension", "blood pressure"]):
                    medical_accuracy_indicators["logical_symptom_diagnosis_pairs"] += 1
                    
            except:
                continue
        
        total_samples = len(samples)
        
        return {
            "medical_accuracy_indicators": {k: v/total_samples for k, v in medical_accuracy_indicators.items()},
            "overall_medical_accuracy": sum(medical_accuracy_indicators.values()) / (len(medical_accuracy_indicators) * total_samples) if total_samples > 0 else 0
        }
    
    def _calculate_quality_metrics(self, samples: List[Dict]) -> Dict:
        """Calculate overall quality metrics"""
        completeness_scores = []
        consistency_scores = []
        
        for sample in samples:
            try:
                output_data = json.loads(sample['output'])
                
                # Completeness score
                required_fields = ['patient_details', 'medical_history', 'current_diagnosis', 'prescription']
                completeness = sum(1 for field in required_fields if field in output_data) / len(required_fields)
                completeness_scores.append(completeness)
                
                # Consistency score (structure consistency)
                if all(field in output_data for field in required_fields):
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.0)
                    
            except:
                completeness_scores.append(0.0)
                consistency_scores.append(0.0)
        
        return {
            "completeness_score": np.mean(completeness_scores) if completeness_scores else 0,
            "consistency_score": np.mean(consistency_scores) if consistency_scores else 0,
            "overall_quality_score": (np.mean(completeness_scores) + np.mean(consistency_scores)) / 2 if completeness_scores else 0
        }
    
    def _calculate_diversity_metrics(self, samples: List[Dict]) -> Dict:
        """Calculate diversity metrics"""
        unique_conversations = set()
        unique_outputs = set()
        topic_diversity = Counter()
        
        for sample in samples:
            # Normalize conversation for comparison
            conversation = re.sub(r'\s+', ' ', sample['input'].lower().strip())
            unique_conversations.add(conversation)
            
            # Normalize output for comparison
            try:
                output_data = json.loads(sample['output'])
                output_str = json.dumps(output_data, sort_keys=True)
                unique_outputs.add(output_str)
            except:
                pass
            
            # Topic diversity
            conversation_lower = sample['input'].lower()
            if any(term in conversation_lower for term in ['blood pressure', 'hypertension']):
                topic_diversity['hypertension'] += 1
            elif any(term in conversation_lower for term in ['diabetes', 'blood sugar']):
                topic_diversity['diabetes'] += 1
            elif any(term in conversation_lower for term in ['asthma', 'breathing']):
                topic_diversity['asthma'] += 1
            else:
                topic_diversity['other'] += 1
        
        total_samples = len(samples)
        
        return {
            "conversation_diversity": len(unique_conversations) / total_samples if total_samples > 0 else 0,
            "output_diversity": len(unique_outputs) / total_samples if total_samples > 0 else 0,
            "topic_distribution": dict(topic_diversity),
            "topic_entropy": self._calculate_entropy(list(topic_diversity.values()))
        }
    
    def _calculate_consistency_metrics(self, samples: List[Dict]) -> Dict:
        """Calculate consistency metrics"""
        structure_patterns = []
        field_presence = defaultdict(list)
        
        for sample in samples:
            try:
                output_data = json.loads(sample['output'])
                
                # Record structure pattern
                structure = tuple(sorted(output_data.keys()))
                structure_patterns.append(structure)
                
                # Record field presence
                for field in ['patient_details', 'medical_history', 'current_diagnosis', 'prescription']:
                    field_presence[field].append(1 if field in output_data else 0)
                    
            except:
                continue
        
        # Calculate consistency scores
        structure_consistency = len(set(structure_patterns)) / len(structure_patterns) if structure_patterns else 0
        field_consistency = {}
        
        for field, presence_list in field_presence.items():
            if presence_list:
                field_consistency[field] = 1 - np.std(presence_list)  # Lower std = higher consistency
        
        return {
            "structure_consistency": structure_consistency,
            "field_consistency": field_consistency,
            "overall_consistency": np.mean(list(field_consistency.values())) if field_consistency else 0
        }
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution"""
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        probabilities = [v/total for v in values if v > 0]
        
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy
    
    def generate_report(self, metrics: Dict = None) -> str:
        """Generate a comprehensive report"""
        if metrics is None:
            metrics = self.metrics
        
        report = []
        report.append("=" * 80)
        report.append("📊 COMPREHENSIVE DATASET EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic Statistics
        basic = metrics['basic_stats']
        report.append("📈 BASIC STATISTICS")
        report.append("-" * 40)
        report.append(f"Total samples: {basic['total_samples']}")
        report.append(f"JSON validity rate: {basic['json_validity_rate']:.2%}")
        report.append(f"Average conversation length: {basic['avg_conversation_length']:.1f} exchanges")
        report.append(f"Conversation length range: {basic['min_conversation_length']} - {basic['max_conversation_length']}")
        report.append(f"Standard deviation: {basic['std_conversation_length']:.2f}")
        report.append("")
        
        # Conversation Analysis
        conv = metrics['conversation_analysis']
        report.append("💬 CONVERSATION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Average doctor turns: {conv['avg_doctor_turns']:.1f}")
        report.append(f"Average patient turns: {conv['avg_patient_turns']:.1f}")
        report.append(f"Conversation balance (patient/doctor): {conv['conversation_balance']:.2f}")
        report.append("")
        
        # Topic Distribution
        report.append("🏥 TOPIC DISTRIBUTION")
        report.append("-" * 40)
        for topic, count in conv['topics_detected'].items():
            percentage = (count / basic['total_samples']) * 100
            report.append(f"{topic.capitalize()}: {count} samples ({percentage:.1f}%)")
        report.append("")
        
        # Quality Metrics
        quality = metrics['quality_metrics']
        report.append("⭐ QUALITY METRICS")
        report.append("-" * 40)
        report.append(f"Completeness score: {quality['completeness_score']:.2%}")
        report.append(f"Consistency score: {quality['consistency_score']:.2%}")
        report.append(f"Overall quality score: {quality['overall_quality_score']:.2%}")
        report.append("")
        
        # Diversity Metrics
        diversity = metrics['diversity_metrics']
        report.append("🌍 DIVERSITY METRICS")
        report.append("-" * 40)
        report.append(f"Conversation diversity: {diversity['conversation_diversity']:.2%}")
        report.append(f"Output diversity: {diversity['output_diversity']:.2%}")
        report.append(f"Topic entropy: {diversity['topic_entropy']:.3f}")
        report.append("")
        
        # Medical Content Analysis
        medical = metrics['medical_content_analysis']
        report.append("🏥 MEDICAL CONTENT ANALYSIS")
        report.append("-" * 40)
        report.append(f"Overall medical accuracy: {medical['overall_medical_accuracy']:.2%}")
        for indicator, score in medical['medical_accuracy_indicators'].items():
            report.append(f"{indicator.replace('_', ' ').title()}: {score:.2%}")
        report.append("")
        
        # Top Medical Terms
        report.append("🔍 TOP MEDICAL TERMS")
        report.append("-" * 40)
        sorted_terms = sorted(conv['medical_terms_frequency'].items(), key=lambda x: x[1], reverse=True)
        for term, count in sorted_terms[:10]:
            percentage = (count / basic['total_samples']) * 100
            report.append(f"{term}: {count} occurrences ({percentage:.1f}%)")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_metrics(self, filepath: str, metrics: Dict = None):
        """Save metrics to JSON file"""
        if metrics is None:
            metrics = self.metrics
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def save_report(self, filepath: str, metrics: Dict = None):
        """Save report to text file"""
        if metrics is None:
            metrics = self.metrics
        
        report = self.generate_report(metrics)
        with open(filepath, 'w') as f:
            f.write(report)

def main():
    """Test the metrics evaluator"""
    # Load test data
    with open("data/test_dataset.jsonl", 'r') as f:
        samples = [json.loads(line.strip()) for line in f]
    
    # Initialize evaluator
    evaluator = MetricsEvaluator()
    
    # Evaluate dataset
    print("🔍 Evaluating dataset...")
    metrics = evaluator.evaluate_dataset(samples)
    
    # Generate and print report
    report = evaluator.generate_report(metrics)
    print(report)
    
    # Save metrics and report
    evaluator.save_metrics("outputs/comprehensive_metrics.json", metrics)
    evaluator.save_report("outputs/comprehensive_report.txt", metrics)
    
    print(f"\n💾 Metrics saved to outputs/comprehensive_metrics.json")
    print(f"📄 Report saved to outputs/comprehensive_report.txt")

if __name__ == "__main__":
    main() 