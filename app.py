"""
app.py

Main Flask entry point for the Face Recognition System web app.

Recognition is embedding-based (ArcFace via DeepFace) — there is no
trained classifier model or retraining step. Registering a new person
just extracts and stores one averaged embedding vector; nothing else
needs to change when someone new signs up.

Routes:
    GET  /                          -> registration page (templates/register.html)
    POST /api/face/register         -> (from register_routes.py) capture + store embedding
    GET  /api/face/list             -> (from register_routes.py) list registered names
    GET  /recognize                 -> live recognition page
    POST /api/face/recognize        -> single-frame recognition
    POST /api/face/verify-and-delete -> the only deletion path; re-verifies at 0.80
                                         similarity via camera before deleting

Folder layout this app expects:
    app.py
    face_embeddings.py
    register_routes.py
    recognition_engine.py
    webcam.py                  (legacy desktop script, classifier-based — not used by Flask)
    Fine_Tune.py                (legacy, retired from the live pipeline)
    augment_frames.py           (legacy, retired from the live pipeline)
    .env                        (local only, NEVER commit — holds MONGODB_URI)
    .env.example                (safe template, committed to git)
    templates/
        register.html
        recognize.html
    static/
        css/register.css, recognize.css
        js/register.js, recognize.js

Storage: MongoDB Atlas (see face_embeddings.py). Requires the
MONGODB_URI environment variable — loaded here from a local .env file
for development; on Hugging Face Spaces, set it as a Secret instead
(Settings -> Variables and secrets), which python-dotenv won't override.
"""

from dotenv import load_dotenv
load_dotenv()  # loads .env into environment variables, if present (local dev only)

import base64
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request

from register_routes import register_blueprint
from recognition_engine import recognize_frame, is_model_ready, load_model, verify_for_deletion
from face_embeddings import delete_embedding

app = Flask(__name__)

# ── Register the face registration/delete/list routes ──
app.register_blueprint(register_blueprint)


# ── Page routes ──

@app.route("/")
def index():
    """Registration page — capture a new person's face."""
    return render_template("register.html")


@app.route("/recognize")
def recognize_page():
    """Live recognition page."""
    return render_template("recognize.html")


# ── Recognition API ──

@app.route("/api/face/recognize", methods=["POST"])
def recognize():
    """
    Expects JSON body: { "frame": "data:image/jpeg;base64,..." }
    Returns the recognition result for that single frame.
    """
    if not is_model_ready():
        return jsonify({
            "error": "No trained model yet. Register at least one person first.",
            "model_ready": False
        }), 409

    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Expected a 'frame' field with a base64 image."}), 400

    try:
        frame_data = data["frame"]
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]

        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image."}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to decode frame: {e}"}), 400

    result = recognize_frame(frame)
    return jsonify(result), 200


@app.route("/api/face/reload-model", methods=["POST"])
def reload_model():
    """
    Re-checks whether anyone is registered. Kept mainly for interface
    consistency — since stored embeddings are read fresh from MongoDB
    on every recognition request, there's no in-memory model that goes
    stale, unlike the old classifier approach.
    """
    success = load_model()
    return jsonify({"success": success, "model_ready": is_model_ready()}), 200


@app.route("/api/face/verify-and-delete", methods=["POST"])
def verify_and_delete():
    """
    Expects JSON body: { "name": "Harshit", "frame": "data:image/jpeg;base64,..." }

    Re-checks the current camera frame against the claimed name at a
    strict 0.80 similarity threshold (separate from the normal 0.60
    recognition threshold). Only deletes if that stricter check passes.
    """
    data = request.get_json(silent=True)
    if not data or "name" not in data or "frame" not in data:
        return jsonify({"error": "Expected 'name' and 'frame' fields."}), 400

    claimed_name = data["name"]

    try:
        frame_data = data["frame"]
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]

        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image."}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to decode frame: {e}"}), 400

    verification = verify_for_deletion(frame, claimed_name)

    if not verification["verified"]:
        return jsonify({
            "deleted": False,
            "similarity": verification["similarity"],
            "error": verification["reason"]
        }), 403

    deleted = delete_embedding(claimed_name)

    if not deleted:
        return jsonify({
            "deleted": False,
            "error": f"Verification passed, but '{claimed_name}' was not found to delete."
        }), 404

    return jsonify({
        "deleted": True,
        "similarity": verification["similarity"],
        "message": f"Deleted all data for '{claimed_name}'."
    }), 200


# ── Health check, useful for confirming the app is running ──

@app.route("/api/health")
def health():
    from face_embeddings import is_anyone_registered
    return jsonify({
        "status": "ok",
        "anyone_registered": is_anyone_registered()
    })


if __name__ == "__main__":
    import os as _os
    # Hugging Face Spaces expects port 7860; PORT env var lets this
    # still run on 5000 (or anything else) for local development.
    port = int(_os.environ.get("PORT", 7860))

    # debug=True is convenient locally (auto-reload, detailed error pages)
    # but should stay off in any real deployment. Controlled via env var
    # so you don't need to edit this file when switching contexts.
    debug_mode = _os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app.run(host="0.0.0.0", port=port, debug=debug_mode)