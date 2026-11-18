# chatbot_app.py - Flask API + web interface for the deep learning intent chatbot
import json
import logging
import os
import random
from datetime import datetime

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from model import LSTMIntentClassifier
from nlp_utils import Vocabulary, tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chatbot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dl_model.pt")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
PORT = int(os.environ.get("CHATBOT_PORT", 5001))
MAX_MESSAGE_CHARS = 500


class IntentChatbot:
    def __init__(self, model_path=MODEL_PATH, intents_path=INTENTS_PATH):
        self.model_path = model_path
        self.intents_path = intents_path
        self.device = torch.device("cpu")
        self.model = None
        self.vocab = None
        self.classes = []
        self.intents = []
        self.max_len = 16
        self.heldout_accuracy = None
        self.confidence_threshold = 0.5

    def load_model(self):
        try:
            ckpt = torch.load(self.model_path, map_location=self.device)
            cfg = ckpt["config"]
            self.max_len = cfg["max_len"]
            self.classes = ckpt["classes"]
            self.vocab = Vocabulary(ckpt["word2idx"])
            self.heldout_accuracy = ckpt.get("heldout_accuracy")
            self.model = LSTMIntentClassifier(
                len(self.vocab), num_classes=len(self.classes),
                **{k: v for k, v in cfg.items() if k != "max_len"})
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            # Responses are read from intents.json, so you can edit replies without retraining.
            with open(self.intents_path, encoding="utf-8") as f:
                self.intents = json.load(f)["intents"]
            logger.info("Model loaded (%d intents, vocab %d)", len(self.classes), len(self.vocab))
            return True
        except Exception as e:
            logger.error("Could not load model: %s", e)
            self.model = None
            return False

    @torch.no_grad()
    def predict_intent(self, message, top_k=3):
        ids = self.vocab.encode(tokenize(message), self.max_len)
        probs = torch.softmax(self.model(torch.tensor([ids])), dim=1)[0]
        values, idx = probs.topk(min(top_k, len(self.classes)))
        return [(self.classes[i], float(v)) for v, i in zip(values.tolist(), idx.tolist())]

    def get_response(self, tag):
        for intent in self.intents:
            if intent["tag"] == tag and intent["responses"]:
                return random.choice(intent["responses"])
        return "I'm not sure I understand. Could you rephrase that?"

    def generate_response(self, message):
        top = self.predict_intent(message)
        intent, confidence = top[0]
        if confidence < self.confidence_threshold:
            intent = "default"
        return {
            "message": self.get_response(intent),
            "intent": intent,
            "confidence": confidence,
            "top_intents": [{"intent": t, "confidence": c} for t, c in top],
            "timestamp": datetime.now().isoformat(),
        }

    def examples(self, per_intent=2):
        out = {}
        for intent in self.intents:
            pats = [p for p in intent["patterns"] if p.strip()]
            if intent["tag"] != "default" and pats:
                out[intent["tag"]] = pats[:per_intent]
        return out


app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"), static_url_path="/static")
CORS(app)
chatbot = IntentChatbot()
if not chatbot.load_model():
    logger.warning("Model not loaded - run: python train_model.py")


def _clean(msg):
    return msg.strip()[:MAX_MESSAGE_CHARS] if isinstance(msg, str) else ""


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


@app.route("/api")
def api_info():
    return jsonify({
        "name": "Intent-Based Chatbot API",
        "version": "2.0.0",
        "model": "BiLSTM + attention (PyTorch)",
        "endpoints": {
            "/": "Web interface",
            "/api": "API information",
            "/chat": "POST - send a message",
            "/batch_chat": "POST - send several messages",
            "/health": "GET - health check",
            "/intents": "GET - list intents",
            "/examples": "GET - example messages per intent",
            "/stats": "GET - model statistics",
        },
    })


@app.route("/health")
def health():
    loaded = chatbot.model is not None
    return jsonify({
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "timestamp": datetime.now().isoformat(),
    }), (200 if loaded else 503)


@app.route("/intents")
def intents():
    if not chatbot.classes:
        return jsonify({"error": "Model not loaded"}), 503
    shown = [c for c in chatbot.classes if c != "default"]
    return jsonify({"intents": shown, "total": len(shown)})


@app.route("/examples")
def examples():
    if not chatbot.intents:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({"examples": chatbot.examples()})


@app.route("/chat", methods=["POST"])
def chat():
    if chatbot.model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    data = request.get_json(silent=True) or {}
    message = _clean(data.get("message"))
    if not message:
        return jsonify({"error": "Provide a non-empty 'message' string"}), 400
    result = chatbot.generate_response(message)
    result["user_message"] = message
    logger.info("User: %s | %s (%.2f)", message, result["intent"], result["confidence"])
    return jsonify(result)


@app.route("/batch_chat", methods=["POST"])
def batch_chat():
    if chatbot.model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    if not isinstance(messages, list):
        return jsonify({"error": "'messages' must be a list of strings"}), 400
    responses = []
    for msg in messages[:100]:
        msg = _clean(msg)
        if msg:
            r = chatbot.generate_response(msg)
            r["user_message"] = msg
            responses.append(r)
    return jsonify({"responses": responses, "total": len(responses)})


@app.route("/stats")
def stats():
    if not chatbot.intents:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({
        "total_intents": len(chatbot.classes),
        "vocab_size": len(chatbot.vocab),
        "confidence_threshold": chatbot.confidence_threshold,
        "heldout_accuracy": chatbot.heldout_accuracy,
        "intents_breakdown": {
            i["tag"]: {"patterns": len([p for p in i["patterns"] if p.strip()]),
                       "responses": len(i["responses"])}
            for i in chatbot.intents
        },
    })


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Internal server error"}), 500


def main():
    if chatbot.model is None:
        print("Model not found. Train it first:  python train_model.py")
        return
    print(f"Chatbot running at http://localhost:{PORT}  (Ctrl+C to stop)")
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
