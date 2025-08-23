"""
FastAPI + Vapi Integration Example

This example shows how to integrate Vapi with FastAPI for high-performance
async web applications with automatic API documentation.
"""

import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from vapi import Vapi
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(title="Vapi FastAPI Example")
client = Vapi(api_key=os.getenv("VAPI_API_KEY"))

# Request/Response models
class CallRequest(BaseModel):
    assistant_id: str
    phone_number: str

class WebhookEvent(BaseModel):
    type: str
    call: dict = None
    transcript: str = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vapi FastAPI Example</title>
    <script src="https://cdn.jsdelivr.net/npm/@vapi-ai/web@latest/dist/vapi.js"></script>
</head>
<body>
    <h1>Vapi Voice Assistant (FastAPI)</h1>
    <button id="startCall">Start Call</button>
    <button id="endCall" disabled>End Call</button>
    <div id="status"></div>
    
    <script>
        const vapi = new Vapi("{vapi_public_key}");
        const assistantId = "{assistant_id}";
        
        document.getElementById('startCall').onclick = async () => {
            try {
                await vapi.start(assistantId);
                document.getElementById('startCall').disabled = true;
                document.getElementById('endCall').disabled = false;
                document.getElementById('status').innerText = 'Call started';
            } catch (error) {
                console.error('Error starting call:', error);
            }
        };
        
        document.getElementById('endCall').onclick = async () => {
            try {
                await vapi.stop();
                document.getElementById('startCall').disabled = false;
                document.getElementById('endCall').disabled = true;
                document.getElementById('status').innerText = 'Call ended';
            } catch (error) {
                console.error('Error ending call:', error);
            }
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page with Vapi web interface"""
    html = HTML_TEMPLATE.format(
        vapi_public_key=os.getenv("VAPI_PUBLIC_KEY", ""),
        assistant_id=os.getenv("VAPI_ASSISTANT_ID", "")
    )
    return HTMLResponse(content=html)

@app.get("/api/assistants")
async def list_assistants():
    """List all assistants"""
    try:
        assistants = client.assistants.list()
        return [{
            "id": a.id,
            "name": a.name,
            "created_at": a.created_at
        } for a in assistants]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/calls")
async def create_call(call_request: CallRequest):
    """Create a new outbound call"""
    try:
        call = client.calls.create(
            assistantId=call_request.assistant_id,
            customer={
                "number": call_request.phone_number
            }
        )
        return {
            "id": call.id,
            "status": call.status,
            "created_at": call.created_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/vapi")
async def vapi_webhook(event: WebhookEvent):
    """Handle Vapi webhooks"""
    print(f"Received Vapi webhook: {event.type}")
    
    if event.type == "call.started":
        # Handle call started
        call_id = event.call.get("id") if event.call else None
        print(f"Call started: {call_id}")
        
    elif event.type == "call.ended":
        # Handle call ended
        call_id = event.call.get("id") if event.call else None
        print(f"Call ended: {call_id}")
        
    elif event.type == "transcript":
        # Handle transcript
        print(f"Transcript: {event.transcript}")
    
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vapi-fastapi-example"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
