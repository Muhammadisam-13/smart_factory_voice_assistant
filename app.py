import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import spacy
import spacy.cli
from pymongo import MongoClient
from gtts import gTTS
import os
import glob
import atexit
import time
from spacy.matcher import Matcher
from datetime import datetime, timedelta

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

# MongoDB setup
try:
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client["smart_factory"]
    sensors = db["sensors"]
    analytics = db["analytics"]
    # Test connection
    client.server_info()
    logger.info("MongoDB connection successful.")
except Exception as e:
    logger.error("MongoDB connection failed: %s", e)
    raise

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

def get_sensor_data(intent, entity_name, entity_type):
    if intent in ["normal_operation", "not_normal", "clogged_filter", "bearing_wear"]:
        filter_map = {
            "normal_operation": {"maintenance": "Normal Operation"},
            "not_normal": {"maintenance": {"$ne": "Normal Operation"}},
            "clogged_filter": {"maintenance": "Clogged Filter"},
            "bearing_wear": {"maintenance": "Bearing Wear"}
        }
        q = sensors.find({"type": "machine", **filter_map[intent]})
        result = [x["name"] for x in q]
        if not result:
            return "No machines found for that status"
        return f"The machines with {intent.replace('_', ' ')} are: {', '.join(result)}"

    if intent in ["cartons_produced", "cartons_sold"]:
        field = "cartons_produced" if intent == "cartons_produced" else "cartons_sold"
        start_date = datetime.now() - timedelta(days=7)
        result = analytics.aggregate([
            {"$match": {field: {"$exists": True}, "DateTime": {"$gte": start_date.isoformat()}}},
            {"$group": {"_id": None, "total": {"$sum": f"${field}"}}}
        ])
        total = next(result, {}).get("total", 0)
        return f"This week, {total} {intent.replace('_', ' ')}" if total else f"No {intent.replace('_', ' ')} data found"

    field, unit = field_map.get(intent, (intent, ""))
    doc = sensors.find_one({"type": entity_type, "name": entity_name})
    if doc and field in doc:
        return f"The {intent.replace('_', ' ')} of the {entity_name} is {doc[field]} {unit}"
    return f"No data found for {entity_name}"

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

# NEW: Root route for basic accessibility
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
