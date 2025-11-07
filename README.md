# Medical AI Copilot API

A Flask-based API that generates SOAP notes from doctor-patient conversations and optional medical images.

## API Endpoints

### 1. Health Check
```
GET /
```
Returns the health status of the service.

### 2. Generate SOAP Notes
```
POST /generate-soap
```

**Request Parameters:**
- `conversation_text` (required): The doctor-patient conversation transcript
- `images` (optional): Medical test images (multiple files supported)
- `api_key` (optional): SambaNova API key (can also be set as environment variable)

**Response Format:**
```json
{
  "status": "success",
  "soap_notes": {
    "subjective": "Chief complaint and history...",
    "objective": "Vital signs and examination findings...",
    "assessment": "Diagnoses and clinical impressions...",
    "plan": "Treatment plan and follow-up..."
  }
}
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following environment variables:
   - `SAMBANOVA_API_KEY`: Your SambaNova API key
4. The build and start commands are automatically detected from `render.yaml`

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the development server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## Expected JSON Response Format

The API returns SOAP notes in the following JSON structure:

```json
{
  "subjective": "Patient reports chest pain after a fall down stairs, sharp and worsening with deep breathing or movement, with difficulty taking deep breaths.",
  "objective": "Tenderness over lower left ribs. Chest X-ray: displaced 7th rib fracture on the left with soft tissue swelling, no pneumothorax.",
  "assessment": "Rib fracture (7th rib on the left).",
  "plan": "Pain management and likely rest; specific follow-up not mentioned."
}
```