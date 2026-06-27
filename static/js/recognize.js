// ── Config ──
const POLL_INTERVAL_MS = 1000

// ── State ──
let mediaStream = null
let pollTimer = null
let isRequestInFlight = false

// ── Elements ──
const video = document.getElementById("webcam-video")
const canvas = document.getElementById("capture-canvas")
const statusText = document.getElementById("status-text")
const noModelBanner = document.getElementById("no-model-banner")
const resultCard = document.getElementById("result-card")
const resultLabel = document.getElementById("result-label")
const resultConfidence = document.getElementById("result-confidence")
const breakdownList = document.getElementById("breakdown-list")
const deleteDataBtn = document.getElementById("delete-data-btn")
const deleteConfirmCard = document.getElementById("delete-confirm-card")
const deleteStatus = document.getElementById("delete-status")
const confirmDeleteBtn = document.getElementById("confirm-delete-btn")
const cancelDeleteBtn = document.getElementById("cancel-delete-btn")

// ── Delete flow state ──
let currentRecognizedName = null
let isDeleteFlowActive = false

// ── Start webcam on page load ──
initWebcam()

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
            statusText.textContent = "Recognizing..."
            startPolling()
        }
    } catch (err) {
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            statusText.textContent = "Camera access denied. Please allow camera access and reload."
        } else if (err.name === "NotFoundError") {
            statusText.textContent = "No camera found on this device."
        } else {
            statusText.textContent = "Camera unavailable."
        }
        console.warn("Webcam error:", err)
    }
}

function startPolling() {
    pollTimer = setInterval(pollOnce, POLL_INTERVAL_MS)
}

function captureFrameAsDataUrl() {
    const ctx = canvas.getContext("2d")
    ctx.save()
    ctx.translate(canvas.width, 0)
    ctx.scale(-1, 1)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    ctx.restore()
    return canvas.toDataURL("image/jpeg", 0.8)
}

async function pollOnce() {
    if (isRequestInFlight || isDeleteFlowActive) return  // skip overlapping requests, or while confirming a delete
    isRequestInFlight = true

    const frameDataUrl = captureFrameAsDataUrl()

    try {
        const response = await fetch("/api/face/recognize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: frameDataUrl })
        })

        const data = await response.json()

        if (response.status === 409 || data.model_ready === false) {
            showNoModelState()
            return
        }

        noModelBanner.style.display = "none"
        renderResult(data)

    } catch (err) {
        console.error("Recognition request failed:", err)
        statusText.textContent = "Connection issue. Retrying..."
    } finally {
        isRequestInFlight = false
    }
}

function showNoModelState() {
    noModelBanner.style.display = "flex"
    resultCard.style.display = "none"
    statusText.textContent = "Waiting for a registered face..."
}

function renderResult(data) {
    const faces = data.faces || []

    if (faces.length === 0) {
        resultCard.style.display = "none"
        currentRecognizedName = null
        statusText.textContent = "No face detected. Make sure your face is visible."
        return
    }

    // Show the first detected face's result
    const face = faces[0]
    resultCard.style.display = "block"
    statusText.textContent = faces.length > 1
        ? `${faces.length} faces detected — showing the first.`
        : "Recognizing..."

    const isUnknown = face.label === "Unknown"
    resultLabel.textContent = face.label
    resultLabel.classList.toggle("unknown", isUnknown)

    const topMatch = face.top_matches[0]
    resultConfidence.textContent = topMatch ? `${(topMatch[1] * 100).toFixed(1)}% confidence` : ""

    renderBreakdown(face.top_matches)

    // Only offer deletion for an actual recognized person, not "Unknown"
    if (isUnknown) {
        currentRecognizedName = null
        deleteDataBtn.style.display = "none"
    } else {
        currentRecognizedName = face.label
        deleteDataBtn.style.display = "block"
        deleteDataBtn.textContent = `That's me (${face.label}) — delete my data`
    }
}

function renderBreakdown(topMatches) {
    breakdownList.innerHTML = ""

    topMatches.forEach(([name, prob]) => {
        const pct = (prob * 100).toFixed(1)

        const row = document.createElement("div")
        row.className = "breakdown-row"
        row.innerHTML = `
            <span class="breakdown-name" title="${name}">${name}</span>
            <div class="breakdown-bar-track">
                <div class="breakdown-bar-fill" style="width: ${pct}%"></div>
            </div>
            <span class="breakdown-pct">${pct}%</span>
        `
        breakdownList.appendChild(row)
    })
}

// ── Cleanup on page leave ──
window.addEventListener("beforeunload", () => {
    if (pollTimer) clearInterval(pollTimer)
    if (mediaStream) mediaStream.getTracks().forEach(track => track.stop())
})

// ── Delete flow ──

deleteDataBtn.addEventListener("click", () => {
    if (!currentRecognizedName) return
    isDeleteFlowActive = true
    resultCard.style.display = "none"
    deleteConfirmCard.style.display = "block"
    deleteStatus.textContent = "Look directly at the camera, then click Confirm Delete."
    deleteStatus.classList.remove("error-text")
})

cancelDeleteBtn.addEventListener("click", () => {
    closeDeleteFlow()
})

confirmDeleteBtn.addEventListener("click", async () => {
    if (!currentRecognizedName) {
        closeDeleteFlow()
        return
    }

    confirmDeleteBtn.disabled = true
    deleteStatus.textContent = "Verifying your face..."
    deleteStatus.classList.remove("error-text")

    const frameDataUrl = captureFrameAsDataUrl()

    try {
        const response = await fetch("/api/face/verify-and-delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: currentRecognizedName, frame: frameDataUrl })
        })

        const data = await response.json()

        if (response.ok && data.deleted) {
            deleteStatus.textContent = `Deleted. (Match confidence: ${(data.similarity * 100).toFixed(1)}%)`
            setTimeout(closeDeleteFlow, 1800)
        } else {
            deleteStatus.textContent = data.error || "Verification failed. Please try again."
            deleteStatus.classList.add("error-text")
        }
    } catch (err) {
        console.error("Delete request failed:", err)
        deleteStatus.textContent = "Connection issue. Please try again."
        deleteStatus.classList.add("error-text")
    } finally {
        confirmDeleteBtn.disabled = false
    }
})

function closeDeleteFlow() {
    isDeleteFlowActive = false
    deleteConfirmCard.style.display = "none"
    currentRecognizedName = null
}