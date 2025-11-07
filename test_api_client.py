"""
Test client for the Medical AI Copilot API
"""

import requests

def test_soap_generation():
    """Test the SOAP notes generation API"""
    
    # API endpoint
    url = "http://localhost:5000/generate-soap"
    
    # Sample conversation text
    conversation_text = """
    Doctor: Good morning, what brings you in today?
    Patient: I've been having chest pain for the past two days.
    Doctor: Can you describe the pain?
    Patient: It's sharp and gets worse when I breathe deeply.
    Doctor: Any other symptoms?
    Patient: I feel short of breath and a bit dizzy.
    Doctor: Let me check your vital signs.
    [Blood pressure: 140/90, Heart rate: 95 bpm]
    Doctor: I'm going to order an ECG and chest X-ray.
    """
    
    # Prepare the payload
    payload = {
        'conversation_text': conversation_text
    }
    
    # Make the request
    try:
        response = requests.post(url, data=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print("\nGenerated SOAP Notes:")
                soap_notes = result['soap_notes']
                print(f"Subjective: {soap_notes['subjective']}")
                print(f"Objective: {soap_notes['objective']}")
                print(f"Assessment: {soap_notes['assessment']}")
                print(f"Plan: {soap_notes['plan']}")
            else:
                print(f"Error: {result['message']}")
        else:
            print(f"HTTP Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    test_soap_generation()