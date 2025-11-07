"""
Script to check available models in SambaNova API
"""

import requests
import json

def check_available_models(api_key):
    """
    Check available models in SambaNova API
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try to get model list
        response = requests.get(
            "https://api.sambanova.ai/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models_data = response.json()
            print("Available models:")
            print(json.dumps(models_data, indent=2))
            return models_data
        else:
            print(f"Error getting models: {response.status_code} - {response.text}")
            
            # Try alternative endpoint
            response = requests.get(
                "https://api.sambanova.ai/v1/meta/models",
                headers=headers
            )
            
            if response.status_code == 200:
                models_data = response.json()
                print("Available models (alternative endpoint):")
                print(json.dumps(models_data, indent=2))
                return models_data
            else:
                print(f"Error getting models (alternative): {response.status_code} - {response.text}")
                return None
            
    except Exception as e:
        print(f"Error checking models: {e}")
        return None

if __name__ == "__main__":
    # Use the API key
    api_key = "1ed95c26-d518-47b5-aa69-217c7bc188aa"
    check_available_models(api_key)