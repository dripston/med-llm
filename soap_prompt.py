"""
Script to generate SOAP notes using LLM with SambaNova
Can handle both conversation text and optional medical images
"""

import requests
import json
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# SOAP notes prompt template
SOAP_PROMPT = """
You are an expert medical scribe assistant. Your task is to generate structured SOAP notes from doctor-patient conversation transcripts and optional medical test images.

SOAP Note Format:
- Subjective (S): Patient's chief complaint, history of present illness, and relevant past medical history
- Objective (O): Measurable data including vital signs, physical examination findings, and test results
- Assessment (A): Clinical impressions, diagnoses, and problem list
- Plan (P): Treatment plan, medications, follow-up instructions, and referrals

Instructions:
- Extract relevant information from the conversation and organize it into the SOAP format
- If medical test images are provided, incorporate relevant findings from the image descriptions
- Be concise but comprehensive
- Use medical terminology appropriately
- If information is not available for a section, mark it as "Not mentioned"
- Prioritize accuracy over completeness

Doctor-Patient Conversation:
{conversation_text}

Medical Test Image Descriptions (if available):
{image_descriptions}

Generate the SOAP notes in the following JSON format:
{{
  "subjective": "Chief complaint and history...",
  "objective": "Vital signs and examination findings...",
  "assessment": "Diagnoses and clinical impressions...",
  "plan": "Treatment plan and follow-up..."
}}

SOAP Notes:
"""

def generate_soap_notes(conversation_text, api_key, image_descriptions=""):
    """
    Generate SOAP notes from conversation text and optional image descriptions using SambaNova LLM
    """
    try:
        # Prepare the prompt
        prompt = SOAP_PROMPT.format(
            conversation_text=conversation_text,
            image_descriptions=image_descriptions if image_descriptions else "No medical test images provided"
        )
        
        # Call SambaNova API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Llama-4-Maverick-17B-128E-Instruct",
            "messages": [
                {"role": "system", "content": "You are a medical scribe assistant that generates SOAP notes from doctor-patient conversations and medical test images."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error generating SOAP notes: {e}")
        return None

def process_medical_images(image_paths, api_key):
    """
    Process medical images and generate descriptions using SambaNova text model
    Since there's no vision model available, we'll simulate the process
    """
    try:
        descriptions = []
        
        for image_path in image_paths:
            # Check if file exists
            if not os.path.exists(image_path):
                descriptions.append(f"Image {image_path}: File not found")
                continue
                
            # Since we don't have a vision model, we'll create a simulated description
            # In a real scenario, this would be replaced with actual image analysis
            simulated_description = """This appears to be a medical test image. Key findings would typically include:
1. Relevant anatomical structures
2. Any abnormal findings
3. Measurements or values
4. Comparison with normal standards"""
            
            descriptions.append(f"Image {os.path.basename(image_path)}: {simulated_description}")
        
        return "\n".join(descriptions)
        
    except Exception as e:
        print(f"Error processing medical images: {e}")
        return ""

# Example usage
if __name__ == "__main__":
    # Test with sample conversation about rib fracture
    sample_conversation = """
    Doctor: Good morning, what brings you in today?
    Patient: I fell down some stairs yesterday and hurt my chest. It's really painful.
    Doctor: Can you describe the pain?
    Patient: It's sharp and gets worse when I breathe deeply or move.
    Doctor: Any difficulty breathing?
    Patient: Yes, it's hard to take deep breaths.
    Doctor: Let me examine your chest.
    [Physical examination: Tenderness over the lower ribs on the left side]
    Doctor: I'm going to order a chest X-ray to check for any fractures.
    """
    
    # Use the API key from environment variables
    api_key = os.environ.get('SAMBANOVA_API_KEY')
    if not api_key:
        print("SAMBANOVA_API_KEY not found in environment variables")
        exit(1)
    
    # Check for chest X-ray image
    image_path = "rib_fracture_big_gallery.jpeg"
    
    # Generate SOAP notes with simulated image processing
    print("Generating SOAP notes with simulated image processing...")
    
    # Simulate image processing
    image_descriptions = """Image rib_fracture_big_gallery.jpeg: This appears to be a chest X-ray showing a rib fracture. Key findings include:
1. Displacement of the 7th rib on the left side
2. Soft tissue swelling around the fracture site
3. No evidence of pneumothorax
4. Normal cardiac silhouette
5. Clear lung fields bilaterally"""
    
    soap_notes = generate_soap_notes(sample_conversation, api_key, image_descriptions)
    
    if soap_notes:
        print("\nGenerated SOAP Notes:")
        print(soap_notes)
    else:
        print("Failed to generate SOAP notes")