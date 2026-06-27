"""
recognition_engine.py

Live recognition logic for the Flask app, using embedding-based
face recognition (ArcFace via DeepFace) instead of the retired
MobileNetV2 classifier approach.

How it works:
    - Detection: DeepFace with MTCNN backend finds faces in a frame.
    - Recognition: each detected face's region is converted into an
      ArcFace embedding (via face_embeddings.get_embedding), then
      compared via cosine similarity against every stored person's
      averaged embedding (face_embeddings.find_best_matches).
    - No model training or loading of .pth weights happens here —
      "the model" is just DeepFace's pretrained ArcFace, used as-is.

Public interface kept identical to the previous classifier-based
version, so app.py and register_routes.py don't need to change:
    - is_model_ready() -> bool
    - load_model() -> bool          (now just re-checks registered people)
    - recognize_frame(frame) -> dict
"""

from deepface import DeepFace

from face_embeddings import (
    get_embedding,
    find_best_matches,
    is_anyone_registered,
)

DETECTOR_BACKEND = "mtcnn"
SIMILARITY_THRESHOLD = 0.60
DELETION_VERIFICATION_THRESHOLD = 0.80


def is_model_ready() -> bool:
    """
    There's no trained model file anymore — "ready" now means
    "at least one person has a stored embedding to compare against."
    """
    return is_anyone_registered()


def load_model() -> bool:
    """
    Kept for interface compatibility with register_routes.py and app.py,
    which call this after registration/retraining to refresh state.
    There's no in-memory model to reload here (stored embeddings are
    read fresh from MongoDB on every comparison via
    face_embeddings.load_all_embeddings), so this just re-checks
    whether anyone is registered.
    """
    return is_anyone_registered()


def recognize_frame(frame) -> dict:
    """
    Runs detection + embedding-based recognition on a single frame
    (BGR numpy array, same format as cv2.VideoCapture().read() returns).

    Returns a dict:
    {
        "faces": [
            {
                "box": {"x": int, "y": int, "w": int, "h": int},
                "label": "Harshit" | "Unknown",
                "top_matches": [["Harshit", 0.91], ["Priya", 0.45], ["Aman", 0.22]]
            },
            ...
        ],
        "model_ready": bool
    }

    Note: top_matches scores here are cosine similarities (roughly 0-1
    for faces in practice), not softmax probabilities — they won't sum
    to 1 across the three entries the way the old classifier's did.
    """
    if not is_anyone_registered():
        return {"faces": [], "model_ready": False}

    try:
        detections = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
    except Exception:
        detections = []

    results = []

    for detection in detections:
        area = detection["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]

        if w <= 0 or h <= 0:
            continue

        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        # get_embedding accepts a numpy array directly (BGR, like cv2 frames)
        query_embedding = get_embedding(face)

        if query_embedding is None:
            # DeepFace's embedding step couldn't process this crop
            # (rare, since extract_faces already found a face here,
            # but can happen on edge cases like extreme blur).
            continue

        top_matches = find_best_matches(query_embedding, top_k=3)

        if not top_matches:
            # Shouldn't happen given the is_anyone_registered() check
            # above, but guard against it defensively.
            continue

        best_label, best_score = top_matches[0]
        label = best_label if best_score >= SIMILARITY_THRESHOLD else "Unknown"

        results.append({
            "box": {"x": x, "y": y, "w": w, "h": h},
            "label": label,
            "top_matches": top_matches
        })

    return {"faces": results, "model_ready": True}


def verify_for_deletion(frame, claimed_name: str) -> dict:
    """
    Stricter, separate check used only when someone requests deletion
    of their own data. Re-detects and re-embeds the current frame, then
    checks specifically how similar it is to the claimed_name's stored
    embedding — requiring a much higher bar (0.80) than normal
    recognition (0.60), since this gates a destructive action.

    Returns:
    {
        "verified": bool,
        "similarity": float | None,
        "reason": str   (only present if verified is False)
    }
    """
    from face_embeddings import load_all_embeddings, cosine_similarity

    all_embeddings = load_all_embeddings()

    if claimed_name not in all_embeddings:
        return {
            "verified": False,
            "similarity": None,
            "reason": f"No registered person named '{claimed_name}'."
        }

    try:
        detections = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
    except Exception:
        detections = []

    if not detections:
        return {
            "verified": False,
            "similarity": None,
            "reason": "No face detected. Make sure your face is clearly visible."
        }

    # Use the largest detected face (most likely the intended subject)
    detections.sort(key=lambda d: d["facial_area"]["w"] * d["facial_area"]["h"], reverse=True)
    area = detections[0]["facial_area"]
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]

    if w <= 0 or h <= 0:
        return {
            "verified": False,
            "similarity": None,
            "reason": "Could not isolate a usable face region."
        }

    face = frame[y:y + h, x:x + w]
    query_embedding = get_embedding(face)

    if query_embedding is None:
        return {
            "verified": False,
            "similarity": None,
            "reason": "Could not process the detected face. Try again with better lighting."
        }

    similarity = cosine_similarity(query_embedding, all_embeddings[claimed_name])

    if similarity >= DELETION_VERIFICATION_THRESHOLD:
        return {"verified": True, "similarity": round(similarity, 4)}

    return {
        "verified": False,
        "similarity": round(similarity, 4),
        "reason": (
            f"Face match confidence ({similarity*100:.1f}%) is below the "
            f"{DELETION_VERIFICATION_THRESHOLD*100:.0f}% required to confirm deletion. "
            f"Look directly at the camera and try again."
        )
    }