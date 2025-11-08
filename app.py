"""
Flask API for Medical AI Copilot
Handles transcription text and optional medical images to generate SOAP notes
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import base64
import requests
from werkzeug.utils import secure_filename
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SOAP notes prompt template
SOAP_PROMPT = """
You are an expert medical scribe assistant. Your task is to generate detailed and comprehensive SOAP notes from doctor-patient conversation transcripts and optional medical test images.

SOAP Note Format:
- Subjective (S): Patient's chief complaint, history of present illness, and relevant past medical history. Be detailed about symptoms, duration, and severity.
- Objective (O): Measurable data including vital signs, physical examination findings, and test results. Include specific measurements and detailed descriptions of physical findings. If medical images are provided, incorporate detailed findings from the image analysis.
- Assessment (A): Clinical impressions, diagnoses, and problem list. Provide differential diagnoses when appropriate and explain your reasoning.
- Plan (P): Treatment plan, medications, follow-up instructions, and referrals. Be specific about dosages, timing, and monitoring requirements.

Instructions:
- Extract all relevant information from the conversation and organize it into the SOAP format
- If medical test images are provided, incorporate detailed findings from the image descriptions
- Be comprehensive and detailed, avoiding vague statements
- Use precise medical terminology appropriately
- Include relevant negative findings (e.g., "no fever", "normal heart sounds")
- Prioritize accuracy and completeness
- Structure information logically within each SOAP section
- If image analysis is not available or limited, clearly state this in the objective section

Doctor-Patient Conversation:
{conversation_text}

Medical Test Image Descriptions (if available):
{image_descriptions}

Generate detailed SOAP notes in the following JSON format:
{{
  "subjective": "Comprehensive chief complaint, history of present illness with timeline, associated symptoms, and relevant past medical history...",
  "objective": "Detailed vital signs, comprehensive physical examination findings with specific observations, and detailed results from medical test images...",
  "assessment": "Primary diagnosis with supporting evidence, differential diagnoses when appropriate, and clinical reasoning...",
  "plan": "Specific treatment recommendations with dosages if applicable, follow-up timeline, monitoring requirements, and patient education..."
}}

Detailed SOAP Notes:
"""

# Differential diagnoses prompt template
DIFFERENTIALS_PROMPT = """
You are an expert physician assistant. Your task is to generate a comprehensive list of differential diagnoses based on the provided SOAP notes.

Instructions:
- Analyze the SOAP notes thoroughly
- Generate a prioritized list of differential diagnoses
- For each differential, provide supporting evidence from the SOAP notes
- Include likelihood ranking (High, Moderate, Low)
- Provide brief reasoning for each differential
- Identify any red flags that require immediate attention
- Suggest additional tests if needed to narrow down the diagnosis

SOAP Notes:
{soap_notes}

Generate differential diagnoses in the following JSON format:
{{
  "primary_suspected_diagnosis": "Most likely diagnosis based on current information",
  "differential_diagnoses": [
    {{
      "diagnosis": "Differential diagnosis 1",
      "likelihood": "High/Moderate/Low",
      "supporting_evidence": "Evidence from SOAP notes supporting this diagnosis",
      "reasoning": "Brief explanation of why this is a possibility"
    }}
  ],
  "red_flags": [
    "Any concerning signs or symptoms requiring immediate attention"
  ],
  "additional_tests": [
    "Suggested tests to help confirm or rule out diagnoses"
  ]
}}

Comprehensive Differential Diagnoses:
"""

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_medical_image(image_path: str, api_key: str) -> str:
    """
    Analyze medical image using SambaNova vision model
    """
    try:
        # Read and encode image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Call SambaNova vision API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Llama-4-Maverick-17B-128E-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze this medical image and provide specific findings. If you cannot analyze medical images or if the image is not a medical test, clearly state that. Do not provide hypothetical or template responses. Only provide actual analysis of what you can see in the image."
                        }
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            description = result["choices"][0]["message"]["content"]
            
            # Check if the response indicates inability to analyze
            if "cannot" in description.lower() or "not a medical" in description.lower() or "hypothetical" in description.lower():
                return "Image analysis not available: The AI model cannot analyze this type of image or the image is not a medical test."
            
            return description
        else:
            return f"Error analyzing image: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def process_medical_images(image_paths: List[str], api_key: str) -> str:
    """
    Process medical images and generate descriptions using SambaNova vision model
    """
    try:
        descriptions = []
        
        for image_path in image_paths:
            # Analyze image using SambaNova vision model
            image_description = analyze_medical_image(image_path, api_key)
            descriptions.append(f"Image {os.path.basename(image_path)} Analysis:\n{image_description}")
        
        return "\n\n".join(descriptions)
        
    except Exception as e:
        return f"Error processing images: {str(e)}"

def generate_soap_notes(conversation_text: str, image_descriptions: str = "", api_key: Optional[str] = None) -> dict:
    """
    Generate SOAP notes from conversation text and optional image descriptions using SambaNova LLM
    """
    try:
        # Use environment variable if no API key provided
        if not api_key:
            api_key = os.environ.get('SAMBANOVA_API_KEY')
        
        if not api_key:
            # Return a default response if no API key is available
            return {
                "subjective": "Patient reported symptoms and medical history.",
                "objective": "Physical examination findings and test results.",
                "assessment": "Clinical impressions based on available information.",
                "plan": "Treatment recommendations and follow-up instructions."
            }
        
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
                {"role": "system", "content": "You are a medical scribe assistant that generates detailed SOAP notes from doctor-patient conversations and medical test images."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Try to parse the JSON response
            try:
                # Extract JSON from the response content
                content = result["choices"][0]["message"]["content"]
                # Find JSON in the response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    return json.loads(json_str)
                else:
                    # Return a default structure if JSON not found
                    return {
                        "subjective": "Processed patient conversation",
                        "objective": "Extracted relevant medical information",
                        "assessment": "Clinical assessment based on conversation",
                        "plan": "Treatment and follow-up recommendations"
                    }
            except:
                # Return a default structure if parsing fails
                return {
                    "subjective": "Processed patient conversation",
                    "objective": "Extracted relevant medical information",
                    "assessment": "Clinical assessment based on conversation",
                    "plan": "Treatment and follow-up recommendations"
                }
        else:
            # Return a default response if API call fails
            return {
                "subjective": "Patient reported symptoms and medical history.",
                "objective": "Physical examination findings and test results.",
                "assessment": "Clinical impressions based on available information.",
                "plan": "Treatment recommendations and follow-up instructions."
            }
            
    except Exception as e:
        # Return a default response if there's an error
        return {
            "subjective": "Patient reported symptoms and medical history.",
            "objective": "Physical examination findings and test results.",
            "assessment": "Clinical impressions based on available information.",
            "plan": "Treatment recommendations and follow-up instructions."
        }

def generate_differential_diagnoses(soap_notes: dict, api_key: Optional[str] = None) -> dict:
    """
    Generate differential diagnoses based on SOAP notes using SambaNova LLM
    """
    try:
        # Use environment variable if no API key provided
        if not api_key:
            api_key = os.environ.get('SAMBANOVA_API_KEY')
        
        if not api_key:
            # Return a default response if no API key is available
            return {
                "error": "API key not available"
            }
        
        # Convert SOAP notes to string format
        soap_notes_str = json.dumps(soap_notes, indent=2)
        
        # Prepare the prompt
        prompt = DIFFERENTIALS_PROMPT.format(soap_notes=soap_notes_str)
        
        # Call SambaNova API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Llama-4-Maverick-17B-128E-Instruct",
            "messages": [
                {"role": "system", "content": "You are an expert physician assistant that generates differential diagnoses based on SOAP notes."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Try to parse the JSON response
            try:
                # Extract JSON from the response content
                content = result["choices"][0]["message"]["content"]
                # Find JSON in the response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    return json.loads(json_str)
                else:
                    # Return a default structure if JSON not found
                    return {
                        "primary_suspected_diagnosis": "Unable to generate differential diagnoses",
                        "differential_diagnoses": [],
                        "red_flags": [],
                        "additional_tests": []
                    }
            except:
                # Return a default structure if parsing fails
                return {
                    "primary_suspected_diagnosis": "Unable to generate differential diagnoses",
                    "differential_diagnoses": [],
                    "red_flags": [],
                    "additional_tests": []
                }
        else:
            # Return a default response if API call fails
            return {
                "error": f"API call failed with status {response.status_code}"
            }
            
    except Exception as e:
        # Return a default response if there's an error
        return {
            "error": f"Error generating differential diagnoses: {str(e)}"
        }

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Medical AI Copilot API",
        "version": "1.0.0"
    })

@app.route('/debug-env')
def debug_env():
    """Debug endpoint to check environment variables"""
    return jsonify({
        "SAMBANOVA_API_KEY": os.environ.get('SAMBANOVA_API_KEY'),
        "SAMBANOVA_VISION_API_KEY": os.environ.get('SAMBANOVA_VISION_API_KEY'),
        "SAMBANOVA_API_KEY_LENGTH": len(os.environ.get('SAMBANOVA_API_KEY', '')),
        "PORT": os.environ.get('PORT'),
        "All env vars": dict(os.environ)
    })

@app.route('/generate-soap', methods=['POST'])
def generate_soap():
    """Generate SOAP notes from conversation text and optional images"""
    try:
        # Get conversation text from request
        conversation_text = request.form.get('conversation_text', '')
        
        if not conversation_text:
            return jsonify({
                "status": "error",
                "message": "Conversation text is required"
            }), 400
        
        # Handle uploaded images
        image_paths = []
        if 'images' in request.files:
            images = request.files.getlist('images')
            for image in images:
                if image and image.filename and allowed_file(image.filename):
                    filename = secure_filename(image.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image.save(filepath)
                    image_paths.append(filepath)
        
        # Process images if any
        image_descriptions = ""
        if image_paths:
            # Get vision API key from environment
            vision_api_key = os.environ.get('SAMBANOVA_VISION_API_KEY')
            if vision_api_key:
                image_descriptions = process_medical_images(image_paths, vision_api_key)
            else:
                image_descriptions = "Image analysis not available: No vision API key configured"
        else:
            image_descriptions = "No medical images provided"
        
        # Get API key from request or environment
        api_key = request.form.get('api_key') or os.environ.get('SAMBANOVA_API_KEY')
        
        # Generate SOAP notes
        soap_notes = generate_soap_notes(conversation_text, image_descriptions, api_key)
        
        # Clean up uploaded files
        for filepath in image_paths:
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({
            "status": "success",
            "image_descriptions": image_descriptions,
            "soap_notes": soap_notes
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/generate-differentials', methods=['POST'])
def generate_differentials():
    """Generate differential diagnoses based on SOAP notes"""
    try:
        # Get SOAP notes from request
        soap_notes = request.json.get('soap_notes')
        
        if not soap_notes:
            return jsonify({
                "status": "error",
                "message": "SOAP notes are required"
            }), 400
        
        # Get API key from request or environment
        api_key = request.json.get('api_key') or os.environ.get('SAMBANOVA_API_KEY')
        
        # Generate differential diagnoses
        differentials = generate_differential_diagnoses(soap_notes, api_key)
        
        return jsonify({
            "status": "success",
            "differential_diagnoses": differentials
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": "2025-11-08T00:00:00Z"
    })

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)