import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import spacy
import spacy.cli
from pymongo import MongoClient # Still imported, but less used for sensor data
from gtts import gTTS
import os
import glob
import atexit
import time
from spacy.matcher import Matcher
from datetime import datetime, timedelta
import requests

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Lazy-load Whisper model using faster-whisper
def get_whisper_model():
    if not hasattr(get_whisper_model, "model"):
        logger.info("Loading faster-whisper 'tiny' model with int8 quantization...")
        get_whisper_model.model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("Faster-whisper model loaded.")
    return get_whisper_model.model

# Ensure spaCy model is downloaded
try:
    spacy.load("en_core_web_sm")
    logger.info("spaCy 'en_core_web_sm' model already loaded.")
except OSError:
    logger.warning("spaCy 'en_core_web_sm' model not found, downloading...")
    spacy.cli.download("en_core_web_sm")
    logger.info("spaCy 'en_core_web_sm' model downloaded.")
nlp = spacy.load("en_core_web_sm")

# MongoDB setup (kept for analytics if still needed, but sensor data will come from external API)
try:
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb+srv://factory:1234@cluster0.t2zyjyl.mongodb.net/SmartFactory")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client["SmartFactory"] 
    # sensors = db["sensors"] # No longer directly used for sensor data
    analytics = db["analytics"] # Keep if you still need analytics data
    # Test connection
    client.server_info()
    logger.info("MongoDB connection successful.")
except Exception as e:
    logger.error("MongoDB connection failed: %s", e)
    # Don't raise if analytics is optional, but for core functionality, keep it.
    # For now, we'll let it pass if only analytics is affected and sensor data comes from external API.
    # raise # Uncomment if MongoDB connection is critical for other parts

# Setup Matcher
matcher = Matcher(nlp.vocab)
machine_names = [
    "Encapsulator", "Labeller", "Cleaner", "Cooler",
    "Cylinder Creator", "Furnace", "Packager", "Bottle Shaper"
]
room_names = ["Machine Room", "Security Room", "Warehouse"]

for name in machine_names:
    matcher.add("MACHINE", [[{"LOWER": w.lower()} for w in name.split()]])
for name in room_names:
    matcher.add("ROOM", [[{"LOWER": w.lower()} for w in name.split()]])

intent_patterns = {
    "temperature": [{"LOWER": "temperature"}],
    "noise": [{"LOWER": {"IN": ["noise", "noise_level"]}}],
    "maintenance": [{"LOWER": "maintenance"}],
    "vibration": [{"LOWER": "vibration"}],
    "power_usage": [{"LOWER": {"IN": ["power", "power_usage"]}}],
    "humidity": [{"LOWER": "humidity"}],
    "smoke": [{"LOWER": "smoke"}],
    "normal_operation": [{"LOWER": {"IN": ["on", "operating", "are"]}}, {"LOWER": {"IN": ["normal", "normally"]}}, {"LOWER": "operation"}],
    "not_normal": [{"LOWER": "not"}, {"LOWER": {"IN": ["normal", "normally", "operation", "operating"]}}, {"LOWER": "operation", "OP": "?"}],
    "clogged_filter": [{"LOWER": {"IN": ["clogged", "clog"]}}, {"LOWER": {"IN": ["filter", "filtr"]}}],
    "bearing_wear": [{"LOWER": "bearing"}, {"LOWER": "wear"}],
    "cartons_produced": [{"LOWER": {"IN": ["cartons", "carton"]}}, {"LOWER": "produced"}],
    "cartons_sold": [{"LOWER": {"IN": ["cartons", "carton"]}}, {"LOWER": "sold"}]
}
for key, pattern in intent_patterns.items():
    matcher.add(key.upper(), [pattern])

def parse_command(text):
    doc = nlp(text.lower())
    intent, entity_name, entity_type = None, None, None
    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        if label in ["MACHINE", "ROOM"]:
            entity_name = doc[start:end].text.title()
            entity_type = "machine" if label == "MACHINE" else "room"
        elif label.lower() in intent_patterns:
            intent = label.lower()
    return intent, entity_name, entity_type

field_map = {
    "temperature": ("temperature", "degrees Celsius"),
    "noise": ("noise_level", "decibels"),
    "maintenance": ("maintenance", ""),
    "vibration": ("vibration", "hertz"),
    "power_usage": ("power_usage", "kilowatts"),
    "humidity": ("humidity", "percent"),
    "smoke": ("smoke", "parts per million")
}

# External API Configuration
EXTERNAL_API_BASE_URL = "https://smart-factory-five.vercel.app"

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

def get_sensor_data(intent, entity_name, entity_type):
    # Fetch data from the external API first
    all_data = _fetch_all_external_data_internal()
    if not all_data:
        return "I'm sorry, I couldn't retrieve the latest factory data."

    rooms_data = all_data.get("rooms", [])
    machines_data = all_data.get("machines", [])
    cartons_num = all_data.get("cartons_num", 0)

    # Handle intents related to machine status (normal_operation, not_normal, clogged_filter, bearing_wear)
    if intent in ["normal_operation", "not_normal", "clogged_filter", "bearing_wear"]:
        matching_machines = []
        for machine in machines_data:
            if "maintenance" in machine:
                if intent == "normal_operation" and machine["maintenance"] == "Normal Operation":
                    matching_machines.append(machine["name"])
                elif intent == "not_normal" and machine["maintenance"] != "Normal Operation":
                    matching_machines.append(machine["name"])
                elif intent == "clogged_filter" and machine["maintenance"] == "Clogged Filter":
                    matching_machines.append(machine["name"])
                elif intent == "bearing_wear" and machine["maintenance"] == "Bearing Wear":
                    matching_machines.append(machine["name"])
        
        if not matching_machines:
            return f"No machines found with {intent.replace('_', ' ')} status."
        return f"The machines with {intent.replace('_', ' ')} are: {', '.join(matching_machines)}."

    # Handle intents related to cartons produced/sold (using data from external API)
    if intent in ["cartons_produced", "cartons_sold"]:
        # The external API provides 'cartons_num' directly, not a time-series for produced/sold.
        # Assuming 'cartons_num' represents a total or current count.
        # If 'cartons_produced' and 'cartons_sold' need historical data,
        # you'd need that from the external API or your MongoDB analytics.
        if intent == "cartons_produced":
            return f"The total number of cartons produced is currently {cartons_num}."
        elif intent == "cartons_sold":
            # The external API doesn't seem to have a 'cartons_sold' field directly.
            # You might need to clarify this with your team or use your own analytics DB.
            return "I can tell you the total cartons, but not specifically cartons sold from this data."
        
    # Handle specific sensor data requests (temperature, noise, humidity, smoke, vibration, power_usage)
    # Search in machines_data first, then rooms_data
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
        field, unit = field_map.get(intent, (intent, ""))
        if field in target_entity:
            return f"The {intent.replace('_', ' ')} of the {target_entity['name']} is {target_entity[field]} {unit}."
        elif intent == "lights" and "lights" in target_entity and entity_type == "room":
             # Special handling for lights in rooms
            light_statuses = ["on" if l else "off" for l in target_entity["lights"]]
            return f"The lights in the {target_entity['name']} are currently {', '.join(light_statuses)}."
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
    intent, entity_name, entity_type = parse_command(text)
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
