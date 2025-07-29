import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
# spacy and spacy.cli are removed
from pymongo import MongoClient # Still imported, but less used for sensor data
from gtts import gTTS
import os
import glob
import atexit
import time
# spacy.matcher is removed
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
    # You might want to raise an exception or use sensible defaults here
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON from {CONFIG_FILE}: {e}")
    # You might want to raise an exception here
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
    analytics = db["analytics"] # Keep if you still need analytics data
    # Test connection
    client.server_info()
    logger.info("MongoDB connection successful.")
except Exception as e:
    logger.error("MongoDB connection failed: %s", e)
    # For this test, we'll allow the app to run even if MongoDB fails,
    # as sensor data comes from the external API.
    # raise # Uncomment if MongoDB connection is critical for other parts

# External API Configuration
EXTERNAL_API_BASE_URL = "https://smartfactoryvoiceassistant-production.up.railway.app/"

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Helper function to fetch data from the external API
def _fetch_all_external_data_internal():
    external_api_url = f"{EXTERNAL_API_BASE_URL}/data/all"
    try:
        logger.info(f"Fetching data from external API: {external_api_url}")
        response = requests.get(external_api_url)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        logger.info("Successfully fetched data from external API.")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from external API: {e}")
        return None # Return None on failure
    except ValueError as e:
        logger.error(f"Error parsing JSON from external API: {e}")
        return None # Return None on failure

# --- NEW parse_command using Gemini LLM ---
def parse_command(text):
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set.")
        return None, None, None # Cannot process without API key

    # Get entities from config for the prompt
    machine_names = config_data.get("entities", {}).get("machines", [])
    room_names = config_data.get("entities", {}).get("rooms", [])
    
    # Define possible intents based on your field_mappings and maintenance_status_map keys
    # This ensures the LLM returns intents your system can handle
    possible_intents = list(config_data.get("field_mappings", {}).keys()) + \
                       list(config_data.get("maintenance_status_map", {}).keys())
    
    prompt = f"""
    You are a factory voice assistant. Your task is to extract the user's intent and any relevant entity (machine or room) from their command.
    
    Here are the possible machine names: {', '.join(machine_names)}
    Here are the possible room names: {', '.join(room_names)}
    Here are the possible intents: {', '.join(possible_intents)}

    Respond ONLY with a JSON object. Do NOT include any other text.
    The JSON object should have the following structure:
    {{
      "intent": "identified_intent_from_list",
      "entity_name": "identified_entity_name_from_list",
      "entity_type": "machine" or "room" or null
    }}
    
    If an intent or entity cannot be identified, use `null` for that field.
    
    Examples:
    User: "What is the temperature of the Furnace?"
    Response: {{"intent": "temperature", "entity_name": "Furnace", "entity_type": "machine"}}

    User: "Is the Encapsulator operating normally?"
    Response: {{"intent": "normal_operation", "entity_name": "Encapsulator", "entity_type": "machine"}}

    User: "Tell me about the noise level in the Machine Room."
    Response: {{"intent": "noise", "entity_name": "Machine Room", "entity_type": "room"}}

    User: "How many cartons produced?"
    Response: {{"intent": "cartons_produced", "entity_name": null, "entity_type": null}}

    User: "Is anything wrong?"
    Response: {{"intent": "not_normal", "entity_name": null, "entity_type": null}}

    User: "What's the status of lights in Warehouse?"
    Response: {{"intent": "lights", "entity_name": "Warehouse", "entity_type": "room"}}

    User: "Hello"
    Response: {{"intent": null, "entity_name": null, "entity_type": null}}

    User: "{text}"
    Response: 
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
                    "entity_type": {"type": "STRING", "enum": ["machine", "room", "null"], "nullable": True}
                },
                "propertyOrdering": ["intent", "entity_name", "entity_type"]
            }
        }
    }

    try:
        logger.info(f"Sending prompt to Gemini API for text: '{text}'")
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        gemini_response = response.json()
        
        # Parse the JSON string from the LLM's text output
        # The LLM returns a string that needs to be parsed as JSON
        llm_output_text = gemini_response.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "{}")
        
        parsed_llm_output = json.loads(llm_output_text)

        intent = parsed_llm_output.get("intent")
        entity_name = parsed_llm_output.get("entity_name")
        entity_type = parsed_llm_output.get("entity_type")
        
        # Ensure entity_name is capitalized correctly if extracted by LLM
        if entity_name:
            entity_name = entity_name.title()

        logger.info(f"Gemini parsed: Intent={intent}, Entity Name={entity_name}, Entity Type={entity_type}")
        return intent, entity_name, entity_type

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        return None, None, None
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.error(f"Error parsing Gemini API response or unexpected format: {e}, Response: {response.text}")
        return None, None, None
# --- END NEW parse_command ---


# Field map from config (used by get_sensor_data)
field_map = config_data.get("field_mappings", {})
maintenance_status_map = config_data.get("maintenance_status_map", {})


def get_sensor_data(intent, entity_name, entity_type):
    # Fetch data from the external API first
    all_data = _fetch_all_external_data_internal()
    if not all_data:
        return "I'm sorry, I couldn't retrieve the latest factory data."

    rooms_data = all_data.get("rooms", [])
    machines_data = all_data.get("machines", [])
    cartons_num = all_data.get("cartons_num", 0)

    # Handle intents related to machine status (normal_operation, not_normal, clogged_filter, bearing_wear)
    if intent in maintenance_status_map: # Use the configurable map
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
        # The external API doesn't seem to have a 'cartons_sold' field directly.
        # You might need to clarify this with your team or use your own analytics DB.
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
        field_info = field_map.get(intent) # Get field_name and unit from config
        if field_info:
            field_name = field_info["field_name"]
            unit = field_info["unit"]
            
            if field_name in target_entity:
                if intent == "lights" and entity_type == "room":
                    light_statuses = ["on" if l else "off" for l in target_entity["lights"]]
                    return f"The lights in the {target_entity['name']} are currently {', '.join(light_statuses)}."
                else:
                    return f"The {intent.replace('_', ' ')} of the {target_entity['name']} is {target_entity[field_name]} {unit}."
        return f"No {intent.replace('_', ' ')} data found for {target_entity['name']}."
    
    return f"No data found for {entity_name}."


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

# Route to directly fetch all data from the external MERN API
# This route is for external clients to call if they need the raw data.
# The internal helper function `_fetch_all_external_data_internal` is used by `get_sensor_data`.
@app.route("/data/fetch_all_external", methods=["GET"])
def fetch_all_external_data():
    data = _fetch_all_external_data_internal()
    if data:
        return jsonify(data)
    return jsonify({"error": "Failed to fetch data from external API."}), 500


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
    logger.info(f"Processing command: {text}")
    intent, entity_name, entity_type = parse_command(text) # This calls the LLM
    if intent:
        response = get_sensor_data(intent, entity_name, entity_type)
        logger.info(f"Generated response: {response}")
        audio_file = text_to_speech(response)
        audio_name = os.path.basename(audio_file) if audio_file else None 
        return jsonify({
            "response": response,
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
