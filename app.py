import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import whisper
import spacy
from pymongo import MongoClient
from gtts import gTTS
import os
import glob
import atexit
import time
from spacy.matcher import Matcher
from datetime import datetime, timedelta

# Lazy-load Whisper model to avoid large image size during build
def get_whisper_model():
    if not hasattr(get_whisper_model, "model"):
        get_whisper_model.model = whisper.load_model("tiny")
    return get_whisper_model.model

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow all origins for now

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Initialize Whisper, spaCy, MongoDB
try:
    logger.debug("Loading Whisper model...")
    model = whisper.load_model("tiny")  # Changed to tiny for free deployment
    logger.debug("Whisper model loaded successfully")
except Exception as e:
    logger.error("Failed to load Whisper model: %s", str(e))
    raise

try:
    logger.debug("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    logger.debug("spaCy model loaded successfully")
except Exception as e:
    logger.error("Failed to load spaCy model: %s", str(e))
    raise

try:
    logger.debug("Connecting to MongoDB...")
    # Use environment variable for MongoDB URI, fallback to localhost for local development
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client["smart_factory"]
    sensors = db["sensors"]
    analytics = db["analytics"]
    logger.debug("MongoDB connection established")
except Exception as e:
    logger.error("Failed to connect to MongoDB: %s", str(e))
    raise

# Initialize spaCy Matcher for entities and intents
matcher = Matcher(nlp.vocab)
machine_names = [
    "Encapsulator", "Labeller", "Cleaner", "Cooler",
    "Cylinder Creator", "Furnace", "Packager", "Bottle Shaper"
]
room_names = ["Machine Room", "Security Room", "Warehouse"]

# Entity patterns
for name in machine_names:
    matcher.add("MACHINE", [[{"LOWER": word.lower()} for word in name.split()]])
for name in room_names:
    matcher.add("ROOM", [[{"LOWER": word.lower()} for word in name.split()]])

# Intent patterns
intent_patterns = {
    "temperature": [{"LOWER": "temperature"}],
    "noise": [{"LOWER": {"IN": ["noise", "noise_level"]}}],
    "maintenance": [{"LOWER": "maintenance"}],
    "vibration": [{"LOWER": "vibration"}],
    "power_usage": [{"LOWER": {"IN": ["power", "power_usage"]}}],
    "humidity": [{"LOWER": "humidity"}],
    "smoke": [{"LOWER": "smoke"}],
    "normal_operation": [
        {"LOWER": {"IN": ["on", "operating", "are"]}},
        {"LOWER": {"IN": ["normal", "normally"]}},
        {"LOWER": "operation"}
    ],
    "not_normal": [
        {"LOWER": "not"},
        {"LOWER": {"IN": ["normal", "normally", "operation", "operating"]}},
        {"LOWER": "operation", "OP": "?"}
    ],
    "clogged_filter": [
        {"LOWER": {"IN": ["clogged", "clog"]}},
        {"LOWER": {"IN": ["filter", "filtr"]}}
    ],
    "bearing_wear": [
        {"LOWER": "bearing"},
        {"LOWER": "wear"}
    ],
    "cartons_produced": [
        {"LOWER": {"IN": ["cartons", "carton"]}},
        {"LOWER": "produced"}
    ],
    "cartons_sold": [
        {"LOWER": {"IN": ["cartons", "carton"]}},
        {"LOWER": "sold"}
    ]
}
for intent, pattern in intent_patterns.items():
    matcher.add(intent.upper(), [pattern])

# Parse command to extract intent and entity
def parse_command(text):
    logger.debug("Parsing command: %s", text)
    doc = nlp(text.lower())
    intent = None
    entity_name = None
    entity_type = None

    # Extract intent and entity using Matcher
    matches = matcher(doc)
    for match_id, start, end in matches:
        match_label = nlp.vocab.strings[match_id]
        if match_label in ["MACHINE", "ROOM"]:
            entity_name = doc[start:end].text.title()
            entity_type = "machine" if match_label == "MACHINE" else "room"
        elif match_label in [k.upper() for k in intent_patterns]:
            intent = match_label.lower()

    # Handle invalid machine names
    if entity_name and entity_name not in machine_names + room_names:
        logger.warning("Invalid entity name: %s", entity_name)
        return None, None, None

    logger.debug("Parsed intent: %s, entity_name: %s, entity_type: %s", intent, entity_name, entity_type)
    return intent, entity_name, entity_type

# Map intent to MongoDB field and response format
field_map = {
    "temperature": ("temperature", "degrees Celsius"),
    "noise": ("noise_level", "decibels"),
    "maintenance": ("maintenance", ""),
    "vibration": ("vibration", "hertz"),
    "power_usage": ("power_usage", "kilowatts"),
    "humidity": ("humidity", "percent"),
    "smoke": ("smoke", "parts per million")
}

# Query MongoDB for machine, room, or analytics data
def get_sensor_data(intent, entity_name, entity_type):
    logger.debug("Querying MongoDB for intent: %s, entity_name: %s, entity_type: %s", intent, entity_name, entity_type)
    
    # Handle aggregate machine queries
    if intent in ["normal_operation", "not_normal", "clogged_filter", "bearing_wear"]:
        if intent == "normal_operation":
            machines = sensors.find({"type": "machine", "maintenance": "Normal Operation"})
            result = [machine["name"] for machine in machines]
            return f"The machines operating normally are: {', '.join(result) or 'None'}" if result else "No machines are operating normally"
        elif intent == "not_normal":
            machines = sensors.find({"type": "machine", "maintenance": {"$ne": "Normal Operation"}})
            result = [machine["name"] for machine in machines]
            return f"The machines not operating normally are: {', '.join(result) or 'None'}" if result else "All machines are operating normally"
        elif intent == "clogged_filter":
            machines = sensors.find({"type": "machine", "maintenance": "Clogged Filter"})
            result = [machine["name"] for machine in machines]
            return f"The machines with a clogged filter are: {', '.join(result) or 'None'}"
        elif intent == "bearing_wear":
            machines = sensors.find({"type": "machine", "maintenance": "Bearing Wear"})
            result = [machine["name"] for machine in machines]
            return f"The machines with bearing wear are: {', '.join(result) or 'None'}"
    
    # Handle analytics queries
    if intent in ["cartons_produced", "cartons_sold"]:
        field = "cartons_produced" if intent == "cartons_produced" else "cartons_sold"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        pipeline = [
            {"$match": {
                field: {"$exists": True},
                "DateTime": {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            }},
            {"$group": {
                "_id": None,
                "total": {"$sum": f"${field}"}
            }}
        ]
        result = analytics.aggregate(pipeline)
        total = next(result, {}).get("total", 0)
        return f"This week, {total} {intent.replace('_', ' ')}" if total else f"No {intent.replace('_', ' ')} data found this week"
    
    # Handle single-field queries
    field, unit = field_map.get(intent, (intent, ""))
    result = sensors.find_one({"type": entity_type, "name": entity_name})
    if result and field in result:
        logger.debug("Found data: %s", result[field])
        value = result[field]
        article = "the" if entity_type == "machine" else "the"
        return f"The {intent.replace('_', ' ')} of {article} {entity_name} is {value} {unit}".strip()
    logger.warning("Data not found for entity_name: %s, entity_type: %s, field: %s", entity_name, entity_type, field)
    return f"No data found for {entity_name}"

# Generate TTS with unique filename using gTTS
def text_to_speech(text):
    logger.debug("Generating TTS for text: %s", text)
    timestamp = int(time.time() * 1000)
    audio_file = f"static/response_{timestamp}.mp3"
    try:
        tts = gTTS(text)
        tts.save(audio_file)
        logger.debug("TTS saved to: %s", audio_file)
        return audio_file
    except Exception as e:
        logger.error("TTS generation failed: %s", str(e))
        return None

# Cleanup function to delete response_*.mp3 files
def cleanup_audio_files():
    if os.path.exists("static"):
        audio_files = glob.glob(os.path.join("static", "response_*.mp3"))
        for audio_file in audio_files:
            try:
                os.remove(audio_file)
                logger.debug("Deleted audio file: %s", audio_file)
            except Exception as e:
                logger.error("Failed to delete audio file %s: %s", audio_file, str(e))
        logger.info("Cleanup completed: all response_*.mp3 files deleted")

# Register cleanup function to run when server exits
atexit.register(cleanup_audio_files)

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Voice Assistant API is running"})

# Transcribe audio
@app.route("/transcribe", methods=["POST"])
def transcribe():
    logger.debug("Received request for /transcribe")
    if "audio" not in request.files:
        logger.error("No audio file provided")
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files["audio"]
    audio_path = "temp.wav"
    try:
        audio_file.save(audio_path)
        logger.debug("Audio file saved to: %s", audio_path)
        model = get_whisper_model()
        result = model.transcribe(audio_path)
        logger.debug("Transcription result: %s", result["text"])
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({"text": result["text"]})
    except Exception as e:
        logger.error("Transcription failed: %s", str(e))
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({"error": str(e)}), 500

# Process command and return response
@app.route("/process_command", methods=["POST"])
def process_command():
    logger.debug("Received request for /process_command")
    data = request.get_json()
    text = data.get("text", "")
    intent, entity_name, entity_type = parse_command(text)
    if intent:
        response_text = get_sensor_data(intent, entity_name, entity_type)
        audio_file = text_to_speech(response_text)
        if audio_file:
            # Return the audio file path that can be accessed via /audio endpoint
            audio_filename = os.path.basename(audio_file)
            logger.debug("Returning response: %s, audio: %s", response_text, audio_filename)
            return jsonify({
                "response": response_text, 
                "audio_filename": audio_filename,
                "audio_url": f"/audio/{audio_filename}"
            })
        else:
            logger.error("Audio file generation failed for response: %s", response_text)
            return jsonify({"response": response_text, "audio_filename": None, "audio_url": None}), 200
    logger.warning("Invalid command: %s", text)
    return jsonify({"error": f"Invalid command or entity: {text}. Try 'temperature of Encapsulator' or 'which machines have clogged filter'."}), 400

# Serve audio files (ADDED BACK)
@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    logger.debug("Serving audio file: %s", filename)
    try:
        return send_file(os.path.join("static", filename), mimetype="audio/mpeg")
    except FileNotFoundError:
        logger.error("Audio file not found: %s", filename)
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Flask server on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
