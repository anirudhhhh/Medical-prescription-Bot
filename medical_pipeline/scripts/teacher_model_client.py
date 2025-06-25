import os
import json
import time
import random
from typing import Dict, List, Optional
import openai
from openai import OpenAI
import anthropic

class TeacherModelClient:
    def __init__(self, model_name: str, config_path: str = "configs/model_configs.json"):
        """
        Initialize teacher model client
        
        Args:
            model_name: Name of the model to use (deepseek, groq, openai, anthropic)
            config_path: Path to configuration file
        """
        self.model_name = model_name
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['teacher_models'][self.model_name]
    
    def _initialize_client(self):
        """Initialize the appropriate API client"""
        api_key = os.getenv(self.config['api_key_env'])
        if not api_key:
            raise ValueError(f"API key not found for {self.model_name}. Set {self.config['api_key_env']}")
        
        if self.model_name in ['deepseek', 'groq', 'openai']:
            return OpenAI(
                api_key=api_key,
                base_url=self.config['api_base']
            )
        elif self.model_name == 'anthropic':
            return anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def generate_conversation(self, topic: str, mood: str, length_range: Dict) -> str:
        """Generate a medical conversation"""
        prompt = self._create_conversation_prompt(topic, mood, length_range)
        return self._call_api(prompt)
    
    def generate_structured_output(self, conversation: str) -> str:
        """Generate structured JSON output from conversation"""
        prompt = self._create_structured_output_prompt(conversation)
        return self._call_api(prompt)
    
    def _create_conversation_prompt(self, topic: str, mood: str, length_range: Dict) -> str:
        """Create prompt for conversation generation"""
        return f"""Generate a realistic doctor-patient conversation about {topic}. 

Requirements:
- Patient mood: {mood}
- Length: {length_range['min_exchanges']}-{length_range['max_exchanges']} exchanges
- Include patient symptoms/concerns
- Show doctor's diagnostic process
- End with a treatment plan
- Make it medically accurate and realistic
- Include patient name, age, and relevant medical history naturally in the conversation

IMPORTANT: Use diverse and natural greetings and conversation styles. Examples:
- "Good morning/afternoon/evening, [Name]. How are you feeling today?"
- "Hello [Name], I'm Dr. [Lastname]. What brings you in today?"
- "Hi there, [Name]. I understand you've been having some concerns about [topic]?"
- "Welcome back, [Name]. How have things been since your last visit?"
- "Hello [Name], I'm [Dr. Lastname]. I see you're here for your [topic] checkup."

Vary the doctor's communication style:
- Some doctors are more formal and clinical
- Some are warm and empathetic
- Some are direct and efficient
- Some ask many follow-up questions
- Some are more conversational and casual

Vary the patient's communication style based on mood:
- Worried patients might be anxious and ask many questions
- Calm patients might be more composed and detailed
- Panicked patients might be breathless or rushed
- Relaxed patients might be casual and chatty
- Concerned patients might be thoughtful and ask for clarification

Format the conversation as:
Doctor: [dialogue]
Patient: [dialogue]
Doctor: [dialogue]
Patient: [dialogue]
... (continue as needed)

Generate only the conversation, no additional text."""

    def _create_structured_output_prompt(self, conversation: str) -> str:
        """Create prompt for structured output generation"""
        return f"""You are a medical assistant that extracts structured information from doctor-patient conversations. 

Extract structured patient information from this doctor-patient conversation:

{conversation}

You must respond with ONLY valid JSON in this exact format, with no additional text, explanations, or formatting (no newlines, no indentation, no extra spaces):

{{"patient_details":{{"name":"patient name or NOT PROVIDED","age":age number or "NOT PROVIDED","gender":"Male/Female or NOT PROVIDED"}},"medical_history":["list of relevant medical history items or NOT PROVIDED"],"current_diagnosis":["list of current diagnoses"],"prescription":[{{"medication":"medication name or instruction","dosage":"dosage if applicable","purpose":"purpose of medication/instruction","instructions":"specific instructions if applicable"}}]}}

Remember: Respond with ONLY the compact JSON object, no formatting, no other text."""

    def _call_api(self, prompt: str) -> str:
        """Make API call with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.model_name == 'anthropic':
                    response = self.client.messages.create(
                        model=self.config['model_name'],
                        max_tokens=self.config['max_tokens'],
                        temperature=self.config['temperature'],
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
                else:
                    # Add system message for better JSON generation
                    messages = [
                        {"role": "system", "content": "You are a helpful medical assistant that generates accurate, structured information in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = self.client.chat.completions.create(
                        model=self.config['model_name'],
                        max_tokens=self.config['max_tokens'],
                        temperature=self.config['temperature'],
                        top_p=self.config['top_p'],
                        messages=messages
                    )
                    return response.choices[0].message.content
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"API call failed, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def validate_json_output(self, json_str: str) -> bool:
        """Validate that the output is valid JSON"""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    def generate_complete_sample(self, topic: str, mood: str, length_range: Dict) -> Optional[Dict]:
        """Generate a complete training sample (conversation + structured output)"""
        try:
            # Generate conversation
            conversation = self.generate_conversation(topic, mood, length_range)
            
            # Generate structured output
            structured_output = self.generate_structured_output(conversation)
            
            # Validate JSON
            if not self.validate_json_output(structured_output):
                print(f"Invalid JSON generated for topic: {topic}")
                return None
            
            return {
                "instruction": "Extract structured patient information (Patient Details, Medical History, Current Diagnosis, Prescription) from the following doctor-patient conversation.",
                "input": conversation,
                "output": structured_output
            }
            
        except Exception as e:
            print(f"Error generating sample for topic {topic}: {e}")
            return None 