"""
register_routes.py

Flask routes for webcam-based face registration, using embedding-based
recognition (ArcFace via DeepFace) instead of the retired MobileNetV2
classifier + Fine_Tune.py retraining approach.

Flow:
    1. Browser captures 45 raw frames, sends them all in one POST
       along with the person's name.
    2. Flask saves the 45 raw frames to a temp folder.
    3. face_embeddings.average_embeddings() extracts an ArcFace embedding
       from each frame and averages them into one vector.
    4. That vector is saved into MongoDB via save_embedding().
    5. Response tells the browser registration succeeded — no training
       step exists anymore, so this is fast (seconds, not minutes).

This module assumes it's imported into your main Flask app, e.g.:

    from register_routes import register_blueprint
    app.register_blueprint(register_blueprint)
"""

import os
import base64
import shutil
import re
from flask import Blueprint, request, jsonify

from face_embeddings import (
    average_embeddings,
    save_embedding,
    load_all_embeddings,
    find_best_matches,
)

register_blueprint = Blueprint("register_blueprint", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_RAW_DIR = os.path.join(BASE_DIR, "_temp_raw_frames")

EXPECTED_FRAME_COUNT = 45
MIN_SUCCESSFUL_EMBEDDINGS = 5  # require at least this many usable frames
DUPLICATE_FACE_THRESHOLD = 0.80  # same bar as deletion verification


def sanitize_name(raw_name: str) -> str:
    """
    Turns a user-provided name into a safe, consistent key.
    e.g. "Harshit Malhotra!" -> "Harshit_Malhotra"
    """
    name = raw_name.strip()
    name = re.sub(r"[^A-Za-z0-9_\- ]", "", name)
    name = name.replace(" ", "_")
    return name


def decode_base64_frame(data_url: str, out_path: str) -> None:
    """
    Decodes a base64 data URL (e.g. 'data:image/jpeg;base64,...')
    and writes it to out_path as a JPEG file.
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    with open(out_path, "wb") as f:
        f.write(img_bytes)


@register_blueprint.route("/api/face/register", methods=["POST"])
def register_face():
    """
    Expects JSON body:
    {
        "name": "Harshit Malhotra",
        "frames": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,...", ...]
    }
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Expected JSON body"}), 400

    raw_name = data.get("name", "")
    frames = data.get("frames", [])

    if not raw_name:
        return jsonify({"error": "Name is required"}), 400

    if not frames:
        return jsonify({"error": "No frames received"}), 400

    if len(frames) < 10:
        return jsonify({
            "error": f"Too few frames received ({len(frames)}). Expected around {EXPECTED_FRAME_COUNT}."
        }), 400

    safe_name = sanitize_name(raw_name)
    if not safe_name:
        return jsonify({"error": "Invalid name provided"}), 400

    existing = load_all_embeddings()
    if safe_name in existing:
        return jsonify({
            "error": f"A person named '{safe_name}' is already registered."
        }), 409

    # ── Step 1: save raw frames to a temp folder ──
    raw_dir = os.path.join(TEMP_RAW_DIR, safe_name)
    os.makedirs(raw_dir, exist_ok=True)

    frame_paths = []

    try:
        for idx, frame_data in enumerate(frames):
            frame_path = os.path.join(raw_dir, f"raw_{idx:03d}.jpg")
            decode_base64_frame(frame_data, frame_path)
            frame_paths.append(frame_path)
    except Exception as e:
        shutil.rmtree(raw_dir, ignore_errors=True)
        return jsonify({"error": f"Failed to decode frames: {e}"}), 400

    # ── Step 2: extract embeddings from each frame, average them ──
    try:
        averaged_vector, num_successful, num_total = average_embeddings(frame_paths)
    except Exception as e:
        shutil.rmtree(raw_dir, ignore_errors=True)
        return jsonify({"error": f"Embedding extraction failed: {e}"}), 500
    finally:
        # Raw temp frames are no longer needed once embeddings are extracted —
        # the averaged vector is all that gets stored, not the photos themselves.
        shutil.rmtree(raw_dir, ignore_errors=True)

    if averaged_vector is None or num_successful < MIN_SUCCESSFUL_EMBEDDINGS:
        return jsonify({
            "error": (
                f"Could not extract enough usable faces from your photos "
                f"({num_successful}/{num_total} succeeded). Try registering again "
                f"with better lighting and your face clearly visible."
            )
        }), 400

    # ── Step 3: check if this face matches someone already registered ──
    # Hard block, no override — one face can only be registered once,
    # regardless of what name is used.
    existing_matches = find_best_matches(averaged_vector, top_k=1)

    if existing_matches:
        closest_name, closest_score = existing_matches[0]

        if closest_score >= DUPLICATE_FACE_THRESHOLD:
            return jsonify({
                "error": (
                    f"This face is already registered as '{closest_name}' "
                    f"({closest_score*100:.1f}% similar). Each person can only "
                    f"register once. If this is a mistake, ask '{closest_name}' "
                    f"to delete their existing registration first."
                )
            }), 409

    # ── Step 4: store the averaged embedding ──
    save_embedding(safe_name, averaged_vector)

    return jsonify({
        "success": True,
        "name": safe_name,
        "frames_used": num_successful,
        "frames_captured": num_total,
        "message": f"Registered '{safe_name}' using {num_successful} of {num_total} captured photos."
    }), 200


@register_blueprint.route("/api/face/list", methods=["GET"])
def list_registered():
    """
    Returns the names of everyone currently registered.
    Useful for a 'who's registered' admin view, or for the
    registration page to warn about duplicate names before submitting.
    """
    names = list(load_all_embeddings().keys())
    return jsonify({"names": names, "count": len(names)}), 200