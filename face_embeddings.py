"""
face_embeddings.py

Embedding-based face recognition core logic, replacing the MobileNetV2
classifier approach (Fine_Tune.py / final_face_model.pth / classes.json
are no longer used by the live pipeline).

How this works:
    Registration:
        - 45 raw captured frames -> each passed through DeepFace (ArcFace
          model) to get a 512-dim embedding vector
        - The 45 embeddings are averaged into ONE vector representing
          that person
        - Stored in MongoDB Atlas (a "registered_faces" collection),
          one document per person: {"name": ..., "embedding": [...512 numbers...]}

    Recognition:
        - A live frame -> DeepFace extracts a face -> ArcFace embedding
        - Compare via cosine similarity against every stored vector
        - Closest match above SIMILARITY_THRESHOLD = recognized person
        - Otherwise -> "Unknown"

No training, no epochs, no retraining when a new person registers —
adding someone is just computing one new vector and saving it.

Storage backend: MongoDB Atlas, via pymongo. Connection string is read
from the MONGODB_URI environment variable — never hardcode it in this
file or commit it to version control.
"""

import os
import numpy as np
from deepface import DeepFace
from pymongo import MongoClient
from pymongo.errors import PyMongoError

DETECTOR_BACKEND = "mtcnn"
EMBEDDING_MODEL = "ArcFace"

# Cosine similarity threshold for a "known" match.
# ArcFace + cosine similarity: same-person pairs typically score above
# ~0.55-0.65; tune this based on real testing with your own captures.
SIMILARITY_THRESHOLD = 0.60

# ── MongoDB connection ──
# Set this as an environment variable, e.g.:
#   export MONGODB_URI="mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/"
# On Hugging Face Spaces, set it under Settings -> Variables and secrets
# (as a Secret, not a public Variable, since it contains a password).
MONGODB_URI = os.environ.get("MONGODB_URI", "")
DB_NAME = "face_recognition_app"
COLLECTION_NAME = "registered_faces"

_client = None
_collection = None


def _get_collection():
    """
    Lazily connects to MongoDB on first use, then reuses the same
    connection for subsequent calls (connecting fresh every time would
    be slow and wasteful).
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    if not MONGODB_URI:
        raise RuntimeError(
            "MONGODB_URI environment variable is not set. "
            "Set it to your MongoDB Atlas connection string before running the app."
        )

    try:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
    except PyMongoError as e:
        _client = None
        raise RuntimeError(
            f"Could not connect to MongoDB. Check your MONGODB_URI is correct "
            f"and your IP is allowed in Atlas Network Access. Details: {e}"
        )

    _collection = _client[DB_NAME][COLLECTION_NAME]
    return _collection


# =========================
# Embedding extraction
# =========================

def get_embedding(image_path_or_array):
    """
    Runs DeepFace's embedding extraction on a single image
    (file path or numpy array, BGR — same as cv2 reads).

    Returns a 1D numpy array (the embedding vector), or None if no
    face could be detected/embedded.
    """
    try:
        result = DeepFace.represent(
            img_path=image_path_or_array,
            model_name=EMBEDDING_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True
        )
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        return None

    if not result:
        return None

    # DeepFace.represent returns a list (one entry per detected face).
    # We expect exactly one face per registration frame.
    return np.array(result[0]["embedding"])


def average_embeddings(frame_paths: list) -> tuple:
    """
    Given a list of image file paths (the 45 raw captured frames),
    extracts an embedding from each and averages them into one vector.

    Returns (averaged_vector: np.ndarray | None, num_successful: int, num_total: int)
    """
    vectors = []

    for path in frame_paths:
        emb = get_embedding(path)
        if emb is not None:
            vectors.append(emb)

    if not vectors:
        return None, 0, len(frame_paths)

    averaged = np.mean(vectors, axis=0)
    return averaged, len(vectors), len(frame_paths)


# =========================
# Storage
# =========================

def load_all_embeddings() -> dict:
    """Returns {"name": np.ndarray, ...} for all registered people."""
    collection = _get_collection()
    documents = collection.find({}, {"name": 1, "embedding": 1})
    return {doc["name"]: np.array(doc["embedding"]) for doc in documents}


def save_embedding(name: str, vector: np.ndarray) -> None:
    """Adds or overwrites one person's stored embedding."""
    collection = _get_collection()
    collection.update_one(
        {"name": name},
        {"$set": {"name": name, "embedding": vector.tolist()}},
        upsert=True
    )


def delete_embedding(name: str) -> bool:
    """Removes one person's stored embedding. Returns True if it existed."""
    collection = _get_collection()
    result = collection.delete_one({"name": name})
    return result.deleted_count > 0


def is_anyone_registered() -> bool:
    collection = _get_collection()
    return collection.count_documents({}, limit=1) > 0


# =========================
# Comparison
# =========================

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def find_best_matches(query_vector: np.ndarray, top_k: int = 3) -> list:
    """
    Compares query_vector against all stored embeddings.

    Returns a list of [name, similarity_score] tuples, sorted descending,
    limited to top_k. Similarity scores are in roughly [-1, 1] (cosine
    similarity), but in practice for faces typically land in [0, 1].
    """
    all_embeddings = load_all_embeddings()

    if not all_embeddings:
        return []

    scored = [
        (name, cosine_similarity(query_vector, vec))
        for name, vec in all_embeddings.items()
    ]

    scored.sort(key=lambda x: x[1], reverse=True)

    return [[name, round(score, 4)] for name, score in scored[:top_k]]