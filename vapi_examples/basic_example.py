"""
Basic Vapi Python SDK Example

This example demonstrates how to use the Vapi Python SDK to manage assistants
and make phone calls programmatically.
"""

import os
from vapi_python import Vapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Vapi client
client = Vapi(api_key=os.getenv("VAPI_API_KEY"))


def list_assistants():
    """List all assistants"""
    try:
        assistants = client.assistants.list()
        print(f"Found {len(assistants)} assistants:")
        for assistant in assistants:
            print(f"  - {assistant.name} (ID: {assistant.id})")
        return assistants
    except Exception as e:
        print(f"Error listing assistants: {e}")
        return []

def create_assistant():
    """Create a new assistant"""
    try:
        assistant = client.assistants.create(
            name="Python SDK Assistant",
            model={
                "provider": "openai",
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant created via the Python SDK."
                    }
                ]
            },
            voice={
                "provider": "elevenlabs",
                "voiceId": "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
            }
        )
        print(f"Created assistant: {assistant.name} (ID: {assistant.id})")
        return assistant
    except Exception as e:
        print(f"Error creating assistant: {e}")
        return None

def make_phone_call(assistant_id, phone_number):
    """Make an outbound phone call"""
    try:
        call = client.calls.create(
            assistantId=assistant_id,
            customer={
                "number": phone_number
            }
        )
        print(f"Call initiated! Call ID: {call.id}")
        print(f"Status: {call.status}")
        return call
    except Exception as e:
        print(f"Error making call: {e}")
        return None

def main():
    print("🚀 Vapi Python SDK Example")
    print("-" * 50)
    
    # List existing assistants
    # print("\n📋 Listing assistants...")
    # assistants = list_assistants()
    
    # Create a new assistant
    print("\n✨ Creating a new assistant...")
    new_assistant = create_assistant()
    
    if new_assistant:
        # Example of making a call (commented out for safety)
        # print("\n📞 Making a phone call...")
        call = make_phone_call(new_assistant.id, "+4108867543")
        print("\n💡 To make a phone call, uncomment the code above and provide a valid phone number")
    
    print("\n✅ Example completed!")

if __name__ == "__main__":
    main()
