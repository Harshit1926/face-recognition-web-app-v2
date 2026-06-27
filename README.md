---
title: Face Recognition Web App v2
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Face Recognition Web App v2

Browser-based face recognition built on face embeddings (ArcFace via DeepFace) instead of a trained classifier. Register your face through the webcam, get recognized live, and delete your own data on demand — with a camera-verified check before deletion actually happens.

This is a rebuild of an earlier MobileNetV2-based face recognizer, originally a desktop script using a closed-set classifier. This version moves recognition entirely into the browser and replaces the classifier with an embedding/similarity approach, so adding a new person never requires retraining anything.

## Features
- Register a face via webcam — captures 45 frames, extracts and averages them into one embedding vector
- Live recognition in the browser — polls the camera once per second, shows the matched name with a confidence breakdown
- Duplicate-face protection — one person can only ever be registered once, even under a different name (hard-blocked, no override)
- Verified self-deletion — deleting your data requires a fresh camera check at a stricter similarity threshold than normal recognition, so no one can delete someone else's data by guessing a name
- No training step — recognition is similarity-based (cosine similarity over ArcFace embeddings), so registering someone new is instant and doesn't affect anyone already registered

## Tech Stack
- **Backend** → Python, Flask, REST API
- **Face detection & embeddings** → DeepFace (MTCNN detector, ArcFace embedding model)
- **Storage** → MongoDB Atlas (one document per registered person: name + averaged embedding vector)
- **Frontend** → HTML, CSS, JavaScript (Web APIs: `getUserMedia` for webcam access)
- **Deployment** → Hugging Face Spaces, Docker

## Architecture Notes
- `face_embeddings.py` — embedding extraction, averaging, MongoDB storage, and cosine similarity comparison
- `register_routes.py` — registration endpoint: receives captured frames, extracts/averages embeddings, checks for duplicate faces, stores the result
- `recognition_engine.py` — live recognition endpoint: detects + embeds a frame, compares against everyone stored, returns the closest matches
- `app.py` — Flask entry point wiring everything together
- Legacy files (`Fine_Tune.py`, `augment_frames.py`, `webcam.py`) are **not used by the live app** — they're kept as a record of the original classifier-based, desktop-only version this project evolved from

## Environment Variables / Secrets
This app requires a MongoDB Atlas connection string to run. Nothing will work without it.

| Variable | Required | Description |
|---|---|---|
| `MONGODB_URI` | Yes | MongoDB Atlas connection string (Drivers → Python format). Password must be URL-encoded if it contains special characters. |
| `PORT` | No | Defaults to `7860` (Hugging Face's expected port). Override for local testing if needed. |
| `FLASK_DEBUG` | No | Defaults to `false`. Set to `true` only for local development. |

**Local development:** create a `.env` file in the project root:
```
MONGODB_URI=mongodb+srv://<username>:<url_encoded_password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
```
This file is git-ignored and never committed — copy `.env.example` as a starting template.

**Hugging Face Spaces deployment:** set `MONGODB_URI` under **Settings → Variables and secrets**, added as a **Secret** (not a public Variable), since it contains your database password.

## MongoDB Setup (one-time)
1. Create a free M0 cluster on [MongoDB Atlas](https://cloud.mongodb.com)
2. Database Access → add a database user (username + password)
3. Network Access → allow `0.0.0.0/0` (required since Hugging Face Spaces doesn't have a fixed outbound IP to whitelist)
4. Connect → Drivers → Python → copy the connection string, substituting in your actual (URL-encoded) password

## Running Locally
```bash
pip install -r requirements.txt
# create .env with your MONGODB_URI, as described above
python app.py
```
Visit `http://127.0.0.1:7860/` to register a face, and `http://127.0.0.1:7860/recognize` for live recognition.

> First run will download DeepFace's MTCNN and ArcFace model weights automatically (a few hundred MB, one-time).

## Deploying to Hugging Face Spaces
1. Create a new Space, SDK: **Docker**
2. Push this repo's contents (the `Dockerfile` here is already configured for port 7860 and the system libraries DeepFace/OpenCV need)
3. Set `MONGODB_URI` as a Secret in the Space's settings
4. The Space builds and starts automatically — give it a minute or two on first load, since importing TensorFlow (a DeepFace dependency) takes longer to initialize than a typical Flask app

## Known Limitations
- No liveness/anti-spoofing check — a photo of a registered person's face could potentially be recognized as them. This is a portfolio/demo project, not a production authentication system.
- Free-tier hosting cold starts can be slow (TensorFlow's import alone can take 15+ seconds in resource-constrained environments).
- Recognition accuracy depends on registration photo quality — consistent lighting and a clearly visible face during the 45-frame capture meaningfully improve results.