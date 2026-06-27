// ── Config ──
const TOTAL_FRAMES = 45
const CAPTURE_INTERVAL_MS = 150  // ~6.75 seconds total to capture all 45 frames

// ── State ──
let personName = ""
let capturedFrames = []
let captureTimer = null
let mediaStream = null

// ── Screen elements ──
const nameScreen = document.getElementById("name-screen")
const captureScreen = document.getElementById("capture-screen")
const processingScreen = document.getElementById("processing-screen")
const resultScreen = document.getElementById("result-screen")

function showScreen(screen) {
    [nameScreen, captureScreen, processingScreen, resultScreen].forEach(s => {
        s.style.display = "none"
    })
    screen.style.display = "block"
}

// ── Webcam elements ──
const video = document.getElementById("webcam-video")
const canvas = document.getElementById("capture-canvas")
const captureStatus = document.getElementById("capture-status")
const progressFill = document.getElementById("progress-fill")
const progressLabel = document.getElementById("progress-label")
const beginBtn = document.getElementById("begin-btn")

// ── Step 1: name entry ──
document.getElementById("start-capture-btn").addEventListener("click", () => {
    const nameInput = document.getElementById("name-input")
    const value = nameInput.value.trim()

    if (!value) {
        alert("Please enter your name.")
        return
    }

    personName = value
    showScreen(captureScreen)
    initWebcam()
})

// ── Step 2: open webcam ──
async function initWebcam() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 480, height: 360 },
            audio: false
        })
        video.srcObject = mediaStream

        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            captureStatus.textContent = "Camera ready. Click \"Start Capturing\" when you're ready."
            beginBtn.disabled = false
        }
    } catch (err) {
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            captureStatus.textContent = "Camera access denied. Please allow camera access and reload."
        } else if (err.name === "NotFoundError") {
            captureStatus.textContent = "No camera found on this device."
        } else {
            captureStatus.textContent = "Camera unavailable."
        }
        console.warn("Webcam error:", err)
    }
}

// ── Step 3: capture loop ──
document.getElementById("begin-btn").addEventListener("click", startCapture)

function startCapture() {
    beginBtn.disabled = true
    capturedFrames = []
    updateProgress(0)
    captureStatus.textContent = "Capturing... slowly turn your head left, right, up, and down."

    captureTimer = setInterval(() => {
        captureFrame()

        if (capturedFrames.length >= TOTAL_FRAMES) {
            clearInterval(captureTimer)
            finishCapture()
        }
    }, CAPTURE_INTERVAL_MS)
}

function captureFrame() {
    const ctx = canvas.getContext("2d")

    // Mirror the frame to match what the user sees (video is mirrored via CSS)
    ctx.save()
    ctx.translate(canvas.width, 0)
    ctx.scale(-1, 1)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    ctx.restore()

    const dataUrl = canvas.toDataURL("image/jpeg", 0.85)
    capturedFrames.push(dataUrl)

    updateProgress(capturedFrames.length)
}

function updateProgress(count) {
    const pct = Math.round((count / TOTAL_FRAMES) * 100)
    progressFill.style.width = pct + "%"
    progressLabel.textContent = `${count} / ${TOTAL_FRAMES}`
}

function finishCapture() {
    captureStatus.textContent = "Capture complete. Submitting..."
    stopWebcam()
    submitRegistration()
}

document.getElementById("cancel-btn").addEventListener("click", () => {
    if (captureTimer) clearInterval(captureTimer)
    stopWebcam()
    resetToStart()
})

function stopWebcam() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop())
        mediaStream = null
    }
}

// ── Step 4: submit to Flask ──
async function submitRegistration() {
    showScreen(processingScreen)

    try {
        const response = await fetch("/api/face/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                name: personName,
                frames: capturedFrames
            })
        })

        const data = await response.json()

        if (!response.ok) {
            showResult(false, data.error || "Registration failed. Please try again.")
            return
        }

        showResult(true, data.message || `Registered ${data.name} successfully.`)
    } catch (err) {
        console.error("Registration request failed:", err)
        showResult(false, "Could not reach the server. Please check your connection and try again.")
    }
}

function showResult(success, message) {
    const icon = document.getElementById("result-icon")
    const heading = document.getElementById("result-heading")
    const messageEl = document.getElementById("result-message")

    if (success) {
        icon.textContent = "✓"
        icon.classList.remove("error")
        heading.textContent = "Registration complete"
    } else {
        icon.textContent = "!"
        icon.classList.add("error")
        heading.textContent = "Registration failed"
    }

    messageEl.textContent = message
    showScreen(resultScreen)
}

document.getElementById("done-btn").addEventListener("click", () => {
    resetToStart()
})

function resetToStart() {
    personName = ""
    capturedFrames = []
    document.getElementById("name-input").value = ""
    updateProgress(0)
    beginBtn.disabled = true
    showScreen(nameScreen)
}