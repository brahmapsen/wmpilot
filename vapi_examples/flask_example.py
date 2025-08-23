"""
Flask + Vapi Integration Example

This example shows how to integrate Vapi with a Flask web application
to handle webhooks and create web-based voice interfaces.
"""

import os
from flask import Flask, request, jsonify, render_template_string
from vapi import Vapi
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = Vapi(api_key=os.getenv("VAPI_API_KEY"))

# Simple HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Vapi Flask Example</title>
    <script src="https://cdn.jsdelivr.net/npm/@vapi-ai/web@latest/dist/vapi.js"></script>
</head>
<body>
    <h1>Vapi Voice Assistant</h1>
    <button id="startCall">Start Call</button>
    <button id="endCall" disabled>End Call</button>
    <div id="status"></div>
    
    <script>
        const vapi = new Vapi("{{ vapi_public_key }}");
        const assistantId = "{{ assistant_id }}";
        
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
        
        vapi.on('message', (message) => {
            console.log('Vapi message:', message);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the main page with Vapi web interface"""
    return render_template_string(
        HTML_TEMPLATE,
        vapi_public_key=os.getenv("VAPI_PUBLIC_KEY"),
        assistant_id=os.getenv("VAPI_ASSISTANT_ID")
    )

@app.route('/webhook/vapi', methods=['POST'])
def vapi_webhook():
    """Handle Vapi webhooks for call events"""
    data = request.json
    event_type = data.get('type')
    
    print(f"Received Vapi webhook: {event_type}")
    
    if event_type == 'call.started':
        call_id = data.get('call', {}).get('id')
        print(f"Call started: {call_id}")
        # Handle call start logic
        
    elif event_type == 'call.ended':
        call_id = data.get('call', {}).get('id')
        print(f"Call ended: {call_id}")
        # Handle call end logic
        
    elif event_type == 'transcript':
        transcript = data.get('transcript')
        print(f"Transcript: {transcript}")
        # Handle transcript
    
    return jsonify({"status": "ok"})

@app.route('/api/assistants', methods=['GET'])
def list_assistants():
    """API endpoint to list all assistants"""
    try:
        assistants = client.assistants.list()
        return jsonify([{
            "id": a.id,
            "name": a.name
        } for a in assistants])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/calls', methods=['POST'])
def create_call():
    """API endpoint to create a new call"""
    try:
        data = request.json
        call = client.calls.create(
            assistantId=data.get('assistantId'),
            customer={
                "number": data.get('phoneNumber')
            }
        )
        return jsonify({
            "id": call.id,
            "status": call.status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
