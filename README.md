# 🤖 AI Face Monitoring Service

> **A standalone FastAPI-based AI service for real-time webcam monitoring using WebSockets.**

This project continuously monitors a trainee's webcam during online training or examinations and detects multiple real-time events such as face presence, unknown persons, multiple faces, head movement, camera freeze, camera blocking, and spoofing attempts.

Built with **FastAPI**, **WebSockets**, **MediaPipe**, **OpenCV**, and optional **ONNX Anti-Spoofing**.

---

## ✨ Features

- 🚀 FastAPI REST & WebSocket service
- 📹 Real-time webcam monitoring
- 😀 Face Detection using MediaPipe
- 👤 Session-based Face Registration & Matching
- 🧠 Head Pose Detection
- 👀 Looking Away Detection
- 👥 Multiple Face Detection
- ❌ Face Missing Detection
- 🔒 Unknown Person Detection
- 📷 Camera Freeze Detection
- 🌑 Camera Block Detection
- 🛡️ Optional AI Anti-Spoof Detection (ONNX)
- ⚡ Low Latency WebSocket Communication
- 🧹 Automatic Session Cleanup
- 📊 Structured JSON Logging
- ⚙️ Environment-based Configuration

---

# 🏗️ Tech Stack

| Technology | Purpose |
|------------|---------|
| FastAPI | Backend API |
| WebSockets | Real-time Communication |
| MediaPipe | Face Detection |
| OpenCV | Image Processing |
| NumPy | Numerical Operations |
| ONNX Runtime *(Optional)* | Anti Spoof Detection |
| face-recognition *(Optional)* | Face Embeddings |
| Uvicorn | ASGI Server |
| Python 3.12+ | Runtime |

---

# 📁 Project Structure

```
face_monitor/
│
├── app.py
├── config.py
├── websocket.py
├── detector.py
├── recognizer.py
├── anti_spoof.py
├── logger.py
├── utils.py
│
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/dev-amit-dots/ai-face-monitoring-service.git

cd ai-face-monitoring-service
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Service

```bash
uvicorn face_monitor.app:app --host 0.0.0.0 --port 8000
```

Server

```
http://localhost:8000
```

---

# ❤️ Health Check

```http
GET /health
```

Example

```bash
curl http://localhost:8000/health
```

---

# 🔌 WebSocket Endpoint

```
ws://localhost:8000/ws/monitor
```

---

# 📤 Send Frame

Send a frame every **1–2 seconds**.

```json
{
  "session_id": "abc123",
  "user_id": "42",
  "timestamp": 1750000000,
  "image": "/9j/4AAQSkZJRgABAQ..."
}
```

The **image** field supports:

- Base64 JPEG
- Base64 PNG
- Browser Data URL

Example

```javascript
socket.send(
  JSON.stringify({
    session_id: sessionId,
    user_id: userId,
    timestamp: Date.now(),
    image: canvas.toDataURL("image/jpeg", 0.75),
  })
);
```

---

# 📥 Response

```json
{
  "session_id": "abc123",
  "status": "FACE_PRESENT",
  "message": "Face detected successfully",
  "timestamp": 1750000000,
  "face_count": 1,
  "head_pose": "screen"
}
```

---

# 📌 Supported Status Codes

| Status | Description |
|---------|-------------|
| ✅ FACE_PRESENT | Face detected |
| ❌ FACE_MISSING | No face detected |
| 👥 MULTIPLE_FACES | More than one face |
| 🚫 UNKNOWN_PERSON | Face does not match registered user |
| 👀 LOOKING_AWAY | User looking away for configured duration |
| 📷 CAMERA_BLOCKED | Camera covered or extremely dark |
| 🧊 CAMERA_FROZEN | Frozen camera frame detected |
| 🛡️ SPOOF_DETECTED | Possible spoof attack detected |
| ⚠️ ERROR | Internal processing error |

---

# ⚙️ Configuration

Create a `.env` file using `.env.example`.

| Variable | Default |
|-----------|----------|
| ENVIRONMENT | production |
| LOG_LEVEL | INFO |
| CORS_ORIGINS | * |
| SESSION_TTL_SECONDS | 300 |
| CLEANUP_INTERVAL_SECONDS | 60 |
| FACE_DETECTION_CONFIDENCE | 0.6 |
| FACE_MATCH_TOLERANCE | 0.55 |
| LOOKING_AWAY_SECONDS | 5.0 |
| FREEZE_SECONDS | 6.0 |
| FREEZE_DIFFERENCE_THRESHOLD | 2.5 |
| DARK_MEAN_THRESHOLD | 18.0 |
| DARK_STD_THRESHOLD | 8.0 |
| SPOOF_MODEL_PATH | unset |
| SPOOF_SCORE_THRESHOLD | 0.7 |

---

# 🔄 Browser Integration

This service only performs AI processing.

Your frontend or backend (Laravel, Django, Node.js, etc.) is responsible for:

- Pausing videos
- Showing warnings
- Ending sessions
- Logging violations
- Displaying notifications

The browser should consume the returned status and implement the desired behavior.

---

# 🎯 Use Cases

- 🎓 Online Learning Platforms
- 📝 Online Examinations
- 👨‍🏫 LMS Systems
- 🏢 Employee Training
- 🆔 Attendance Monitoring
- 🛡️ AI Proctoring
- 📹 Webcam Verification

---

# 🛣️ Roadmap

- Face Landmark Tracking
- Eye Blink Detection
- Mouth Open Detection
- Emotion Recognition
- YOLO Person Detection
- Mobile Support
- Docker Support
- Kubernetes Deployment
- Prometheus Metrics
- Grafana Dashboard

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push your branch
5. Open a Pull Request

---

# 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

# ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub.

It helps the project grow and encourages future development.

---

Made with ❤️ using **FastAPI**, **MediaPipe**, **OpenCV**, and **Python**.
