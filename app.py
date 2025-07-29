import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from pymongo import MongoClient
from gtts import gTTS
import os
import glob
import atexit
import time
from datetime import datetime, timedelta
import requests
import json 

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# --- Load Configuration from config.json ---
CONFIG_FILE = 'config.json'
config_data = {}
try:
    with open(CONFIG_FILE, 'r') as f:
        config_data = json.load(f)
    logger.info(f"Configuration loaded from {CONFIG_FILE}")
except FileNotFoundError:
    logger.error(f"Configuration file {CONFIG_FILE} not found. Using default empty config.")
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON from {CONFIG_FILE}: {e}")
# --- End Configuration Load ---

# Lazy-load Whisper model using faster-whisper
def get_whisper_model():
    if not hasattr(get_whisper_model, "model"):
        logger.info("Loading faster-whisper 'tiny' model with int8 quantization...")
        get_whisper_model.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("Faster-whisper model loaded.")
    return get_whisper_model.model

# MongoDB setup (kept for analytics if still needed, but sensor data will come from external API)
try:
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb+srv://factory:1234@cluster0.t2zyjyl.mongodb.net/SmartFactory")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client["SmartFactory"] 
    analytics = db["analytics"]
    client.server_info()
    logger.info("MongoDB connection successful.")
except Exception as e:
    logger.error("MongoDB connection failed: %s", e)

# External MERN API Configuration
EXTERNAL_API_BASE_URL = "https://smart-factory-five.vercel.app" # Updated to the MERN API URL

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Helper function to fetch data from the external API (GET /data/all)
# This function NO LONGER ACCEPTS a user_auth_token and does not send an Authorization header.
# This assumes the /data/all endpoint on the MERN API is publicly accessible.
def _fetch_all_external_data_internal():
    external_api_url = f"{EXTERNAL_API_BASE_URL}/data/all"
    
    try:
        logger.info(f"Fetching data from external API: {external_api_url} (no auth required for this endpoint)")
        response = requests.get(external_api_url) 
        response.raise_for_status() 
        data = response.json()
        logger.info("Successfully fetched data from external API.")
        return data
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching data from external API: {e.response.status_code} - {e.response.text}")
        # Note: No 401/403 specific handling here, as it's assumed to be public.
        return {"error": f"Failed to retrieve data: {e.response.text}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data from external API: {e}")
        return {"error": "Network error retrieving data from factory system."}
    except ValueError as e:
        logger.error(f"Error parsing JSON from external API: {e}")
        return {"error": "Error parsing data from factory system."}

# --- parse_command using Gemini LLM (Expanded for Actions and parameters) ---
def parse_command(text):
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set.")
        return None, None, None, None, None, None, None

    # Get entities from config for the prompt
    machine_names = config_data.get("entities", {}).get("machines", [])
    room_names = config_data.get("entities", {}).get("rooms", [])
    
    # Define all possible intents from config
    possible_intents = list(config_data.get("field_mappings", {}).keys()) + \
                       list(config_data.get("maintenance_status_map", {}).keys()) + \
                       ["toggle_lights", "toggle_machine_power", "record_sale", "record_cartons"] 
    
    prompt = f"""
    You are an advanced AI assistant for a smart factory. Your primary function is to accurately parse natural language commands from users to identify their core intent and any specific factory entity (machine or room) they are referring to, as well as any numerical or specific string parameters required for actions.

    **Constraints & Output Format:**
    - Your response MUST be a single JSON object. DO NOT include any additional text, explanations, or conversational filler outside the JSON.
    - The JSON structure MUST contain the keys "intent", "entity_name", "entity_type", "light_num", "cartons_sold", "cartons_produced", and "buyer".
    - "intent" must be one of the provided 'Available Intents'.
    - "entity_name" must be one of the provided 'Available Entities' (machine or room name), exactly as listed (case-insensitive matching for identification, but output should be Title Case if found).
    - "entity_type" must be "machine", "room", or null.
    - "light_num" must be an integer (1 or 2 for the two lights in a room, or null).
    - "cartons_sold" must be an integer (number of cartons sold, or null).
    - "cartons_produced" must be an integer (number of cartons produced, or null).
    - "buyer" must be a string (name of the buyer, or null).
    - If a parameter is not relevant to the identified intent, its value MUST be `null`.
    - If an intent or entity cannot be confidently identified from the provided lists, its value MUST be `null`.
    - If multiple entities are mentioned, identify the primary one or null if ambiguous.

    **Available Entities:**
    Machines: {', '.join(machine_names)}
    Rooms: {', '.join(room_names)}

    **Available Intents:**
    {', '.join(possible_intents)}

    ---

    **Examples (User Input -> JSON Output):**

    1. User: "What's the current temperature in the Furnace?"
       Output: {{"intent": "temperature", "entity_name": "Furnace", "entity_type": "machine", "light_num": null, "cartons_sold": null, "cartons_produced": null, "buyer": null}}

    2. User: "Is the Encapsulator having any issues with maintenance?"
       Output: {{"intent": "maintenance", "entity_name": "Encapsulator", "entity_type": "machine", "light_num": null, "cartons_sold": null, "cartons_produced": null, "buyer": null}}

    3. User: "Turn on light number one in the Machine Room."
       Output: {{"intent": "toggle_lights", "entity_name": "Machine Room", "entity_type": "room", "light_num": 1, "cartons_sold": null, "cartons_produced": null, "buyer": null}}

    4. User: "Switch off the second light in the Warehouse."
       Output: {{"intent": "toggle_lights", "entity_name": "Warehouse", "entity_type": "room", "light_num": 2, "cartons_sold": null, "cartons_produced": null, "buyer": null}}

    5. User: "Start the Cooler."
       Output: {{"intent": "toggle_machine_power", "entity_name": "Cooler", "entity_type": "machine", "light_num": null, "cartons_sold": null, "cartons_produced": null, "buyer": null}}

    6. User: "Record a sale of 50 cartons to John Doe."
       Output: {{"intent": "record_sale", "entity_name": null, "entity_type": null, "light_num": null, "cartons_sold": 50, "cartons_produced": null, "buyer": "John Doe"}}

    7. User: "Log 100 produced cartons."
       Output: {{"intent": "record_cartons", "entity_name": null, "entity_type": null, "light_num": null, "cartons_sold": null, "cartons_produced": 100, "buyer": null}}

    8. User: "What's up with the Cleaner?"
       Output: {{"intent": "not_normal", "entity_name": "Cleaner", "entity_type": "machine", "light_num": null, "cartons_sold": null, "cartons_produced": null, "buyer": null}}

    9. User: "Hello, how are you?"
       Output: {{"intent": null, "entity_name": null, "entity_type": null, "light_num": null, "cartons_sold": null, "cartons_produced": null, "buyer": null}}
    ---

    **User Command:** "{text}"
    **Your JSON Output:**
    """

    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "intent": {"type": "STRING", "nullable": True},
                    "entity_name": {"type": "STRING", "nullable": True},
                    "entity_type": {"type": "STRING", "enum": ["machine", "room", "null"], "nullable": True},
                    "light_num": {"type": "INTEGER", "nullable": True},
                    "cartons_sold": {"type": "INTEGER", "nullable": True},
                    "cartons_produced": {"type": "INTEGER", "nullable": True},
                    "buyer": {"type": "STRING", "nullable": True}
                },
                "propertyOrdering": ["intent", "entity_name", "entity_type", "light_num", "cartons_sold", "cartons_produced", "buyer"]
            }
        }
    }

    try:
        logger.info(f"Sending prompt to Gemini API for text: '{text}'")
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload)
        response.raise_for_status()
        
        gemini_response = response.json()
        
        llm_output_text = gemini_response.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "{}")
        
        parsed_llm_output = json.loads(llm_output_text)

        intent = parsed_llm_output.get("intent")
        entity_name = parsed_llm_output.get("entity_name")
        entity_type = parsed_llm_output.get("entity_type")
        light_num = parsed_llm_output.get("light_num")
        cartons_sold = parsed_llm_output.get("cartons_sold")
        cartons_produced = parsed_llm_output.get("cartons_produced")
        buyer = parsed_llm_output.get("buyer")
        
        if entity_name:
            entity_name = entity_name.title()

        logger.info(f"Gemini parsed: Intent={intent}, Entity Name={entity_name}, Entity Type={entity_type}, Light Num={light_num}, Cartons Sold={cartons_sold}, Cartons Produced={cartons_produced}, Buyer={buyer}")
        return intent, entity_name, entity_type, light_num, cartons_sold, cartons_produced, buyer

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        return None, None, None, None, None, None, None
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.error(f"Error parsing Gemini API response or unexpected format: {e}, Response: {response.text}")
        return None, None, None, None, None, None, None

# Field map from config (used by get_sensor_data)
field_map = config_data.get("field_mappings", {})
maintenance_status_map = config_data.get("maintenance_status_map", {})


# get_sensor_data NO LONGER ACCEPTS user_auth_token, as /data/all is assumed public
def get_sensor_data(intent, entity_name, entity_type):
    # Fetch data from the external API first (no auth token passed here)
    all_data = _fetch_all_external_data_internal()
    
    # Check if _fetch_all_external_data_internal returned an error dictionary
    if isinstance(all_data, dict) and "error" in all_data:
        return all_data["error"] # Return the error message directly

    if not all_data: # If it's None (e.g., network error)
        return "I'm sorry, I couldn't retrieve the latest factory data."

    rooms_data = all_data.get("rooms", [])
    machines_data = all_data.get("machines", [])
    cartons_num = all_data.get("cartons_num", 0)

    # Handle intents related to machine status (normal_operation, not_normal, clogged_filter, bearing_wear)
    if intent in maintenance_status_map:
        matching_machines = []
        target_maintenance_status = maintenance_status_map[intent]
        
        for machine in machines_data:
            if "maintenance" in machine:
                if isinstance(target_maintenance_status, dict) and "$ne" in target_maintenance_status:
                    if machine["maintenance"] != target_maintenance_status["$ne"]:
                        matching_machines.append(machine["name"])
                elif machine["maintenance"] == target_maintenance_status:
                    matching_machines.append(machine["name"])
        
        if not matching_machines:
            return f"No machines found with {intent.replace('_', ' ')} status."
        return f"The machines with {intent.replace('_', ' ')} are: {', '.join(matching_machines)}."

    # Handle intents related to cartons produced/sold (using data from external API)
    if intent == "cartons_produced":
        return f"The total number of cartons produced is currently {cartons_num}."
    elif intent == "cartons_sold":
        return "I can tell you the total cartons, but not specifically cartons sold from this data."
        
    # Handle specific sensor data requests (temperature, noise, humidity, smoke, vibration, power_usage, lights)
    target_entity = None
    if entity_type == "machine":
        for machine in machines_data:
            if machine["name"].lower() == entity_name.lower():
                target_entity = machine
                break
    elif entity_type == "room":
        for room in rooms_data:
            if room["name"].lower() == entity_name.lower():
                target_entity = room
                break

    if target_entity:
        field_info = field_map.get(intent)
        if field_info:
            field_name = field_info["field_name"]
            unit = field_info["unit"]
            
            if field_name in target_entity:
                if intent == "lights" and entity_type == "room":
                    light_statuses = ["off", "on"]
                    current_light_states = [light_statuses[int(l)] for l in target_entity["lights"]]
                    return f"The lights in the {target_entity['name']} are currently light one is {current_light_states[0]} and light two is {current_light_states[1]}."
                else:
                    return f"The {intent.replace('_', ' ')} of the {target_entity['name']} is {target_entity[field_name]} {unit}."
        return f"No {intent.replace('_', ' ')} data found for {target_entity['name']}."
    
    return f"No data found for {entity_name}."

# Function to perform actions via external API POST requests
# This function still requires and uses user_auth_token
async def perform_action(intent, entity_name, entity_type, light_num, cartons_sold, cartons_produced, buyer, user_auth_token):
    if not user_auth_token:
        logger.warning("No authentication token provided for action request.")
        return "I'm sorry, I need you to log in first to perform this action."

    headers = {
        "Authorization": f"Bearer {user_auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        if intent == "toggle_lights":
            if not entity_name or not entity_type == "room" or light_num is None:
                return "I need a room name and the light number (1 or 2) to toggle lights."
            if light_num not in [1, 2]:
                return "Light number must be 1 or 2."
            
            payload = {
                "room_name": entity_name,
                "light_num": light_num
            }
            api_endpoint = f"{EXTERNAL_API_BASE_URL}/toggle/lights"
            logger.info(f"Toggling lights: {payload} at {api_endpoint}")
            response = requests.post(api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            light_statuses = ["off", "on"]
            current_light_state = light_statuses[int(result['lights'][light_num-1])]
            return f"Light {light_num} in {result.get('room_name')} is now {current_light_state}."

        elif intent == "toggle_machine_power":
            if not entity_name or not entity_type == "machine":
                return "I need a machine name to toggle its power."
            payload = {
                "machine_name": entity_name
            }
            api_endpoint = f"{EXTERNAL_API_BASE_URL}/toggle/machine"
            logger.info(f"Toggling machine power: {payload} at {api_endpoint}")
            response = requests.post(api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            power_status = "on" if result.get('power') else "off"
            return f"The {result.get('machine_name')} is now {power_status}."

        elif intent == "record_sale":
            if cartons_sold is None or not isinstance(cartons_sold, int) or cartons_sold <= 0:
                return "Please specify a valid number of cartons sold."
            
            payload = {
                "cartons_sold": cartons_sold,
                "DateTime": datetime.now().isoformat(),
                "Buyer": buyer if buyer else "Unknown"
            }
            api_endpoint = f"{EXTERNAL_API_BASE_URL}/tx/sale"
            logger.info(f"Recording sale: {payload} at {api_endpoint}")
            response = requests.post(api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return f"Recorded sale of {result.get('cartons_sold')} cartons to {result.get('Buyer')}."

        elif intent == "record_cartons":
            if cartons_produced is None or not isinstance(cartons_produced, int) or cartons_produced <= 0:
                return "Please specify a valid number of cartons produced."
            payload = {
                "cartons_produced": cartons_produced,
                "DateTime": datetime.now().isoformat()
            }
            api_endpoint = f"{EXTERNAL_API_BASE_URL}/tx/cartons"
            logger.info(f"Recording cartons produced: {payload} at {api_endpoint}")
            response = requests.post(api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return f"Recorded production of {result.get('addition', {}).get('cartons_produced')} cartons. Total cartons now {result.get('cartons_num')}."

        else:
            return "I'm sorry, I don't know how to perform that specific action yet."

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error performing action for {intent}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 401 or e.response.status_code == 403:
            return "Authentication failed. Please ensure you are logged in."
        elif e.response.status_code == 400:
            return f"Bad request: {e.response.json().get('error', e.response.text)}"
        elif e.response.status_code == 404:
            return f"Entity not found: {e.response.json().get('error', e.response.text)}"
        return f"Failed to perform action due to a server error: {e.response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error performing action for {intent}: {e}")
        return "I'm sorry, I'm having trouble connecting to the factory system."
    except Exception as e:
        logger.error(f"Unexpected error performing action for {intent}: {e}")
        return "An unexpected error occurred while trying to perform that action."


def text_to_speech(text):
    try:
        ts = int(time.time() * 1000)
        filename = f"static/response_{ts}.mp3"
        gTTS(text).save(filename)
        return filename
    except Exception as e:
        logger.error("TTS error: %s", e)
        return None

def cleanup_audio_files():
    for f in glob.glob("static/response_*.mp3"):
        try:
            os.remove(f)
        except Exception as e:
            logger.error(f"Error removing audio file {f}: {e}")

atexit.register(cleanup_audio_files)

@app.route("/", methods=["GET"])
def index():
    return "Smart Factory Voice Assistant API is running! Use /transcribe or /process_command endpoints."

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    path = "temp.wav"
    try:
        request.files["audio"].save(path)
        model = get_whisper_model()
        segments, info = model.transcribe(path)
        transcribed_text = " ".join([segment.text for segment in segments])
        
        return jsonify({"text": transcribed_text})
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.route("/process_command", methods=["POST"])
def process_command():
    text = request.get_json().get("text", "")
    # Extract token from the request headers sent by the frontend
    # This token is now the user's JWT from the MERN API
    user_auth_token = request.headers.get('Authorization', '').replace('Bearer ', '')

    logger.info(f"Processing command: {text}")
    logger.info(f"Received user_auth_token: {'Present' if user_auth_token else 'Absent'}")
    
    # Parse command using LLM, returns all extracted parameters
    intent, entity_name, entity_type, light_num, cartons_sold, cartons_produced, buyer = parse_command(text) 
    
    response_text = "I'm sorry, I couldn't understand your command or extract enough information."

    if intent:
        action_intents = ["toggle_lights", "toggle_machine_power", "record_sale", "record_cartons"]
        
        if intent in action_intents:
            # Pass the user_auth_token to the perform_action function
            response_text = await perform_action(intent, entity_name, entity_type, light_num, cartons_sold, cartons_produced, buyer, user_auth_token)
        else:
            # DO NOT pass the user_auth_token to the get_sensor_data function
            # as /data/all is assumed to be public.
            response_text = get_sensor_data(intent, entity_name, entity_type)
        
        logger.info(f"Generated response: {response_text}")
        audio_file = text_to_speech(response_text)
        audio_name = os.path.basename(audio_file) if audio_file else None 
        return jsonify({
            "response": response_text,
            "audio_filename": audio_name,
            "audio_url": f"/audio/{audio_name}" if audio_name else None
        })
    logger.warning(f"Could not understand command: {text}")
    return jsonify({"error": f"Could not understand: {text}"}), 400

@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    try:
        return send_file(os.path.join("static", filename), mimetype="audio/mpeg")
    except FileNotFoundError:
        logger.error(f"Audio file not found: {filename}")
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)
