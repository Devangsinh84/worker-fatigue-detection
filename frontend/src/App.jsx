import { useRef, useEffect, useState, useCallback } from "react"
import { AlertTriangle, Eye, EyeOff, Activity, Wifi, WifiOff } from "lucide-react"

const CLOSED_FRAMES_LIMIT = 30

export default function App() {
  const videoRef        = useRef(null)
  const canvasRef       = useRef(null)
  const overlayRef      = useRef(null)
  const intervalRef     = useRef(null)
  const trackingRef     = useRef(null)
  const detectorRef     = useRef(null)
  const isRunningRef    = useRef(false)
  const closedFramesRef = useRef(0)

  const [running, setRunning]           = useState(false)
  const [status, setStatus]             = useState("idle")
  const [ear, setEar]                   = useState(0)
  const [cnnScore, setCnnScore]         = useState(0)
  const [closedFrames, setClosedFrames] = useState(0)
  const [fatigue, setFatigue]           = useState(false)
  const [faceDetected, setFaceDetected] = useState(false)
  const [apiOnline, setApiOnline]       = useState(false)
  const [showTracking, setShowTracking] = useState(true)
  const [earThreshold, setEarThreshold] = useState(0.204)
  const [faceBox, setFaceBox] = useState(null)

  useEffect(() => {
    fetch("http://localhost:8000")
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false))
  }, [])

  const drawTrackingBox = useCallback(() => {
    if (!overlayRef.current || !videoRef.current) return
    const overlay = overlayRef.current
    const ctx     = overlay.getContext("2d")

    overlay.width  = videoRef.current.videoWidth  || overlay.offsetWidth
    overlay.height = videoRef.current.videoHeight || overlay.offsetHeight

    ctx.clearRect(0, 0, overlay.width, overlay.height)

    if (!showTracking || !isRunningRef.current || !faceBox) return

    const W = overlay.width
    const H = overlay.height

    const pad = 0.03
    const x   = (faceBox.x - pad) * W
    const y   = (faceBox.y - pad) * H
    const w   = (faceBox.width  + pad * 2) * W
    const h   = (faceBox.height + pad * 2) * H

    const color  = fatigue             ? "#ef4444" :
                  status === "closed" ? "#eab308" : "#22c55e"
    const now    = Date.now()
    const pulse  = 0.6 + 0.4 * Math.sin(now / 300)
    const corner = 22

    ctx.strokeStyle = color
    ctx.lineWidth   = 2.5
    ctx.globalAlpha = pulse
    ctx.lineCap     = "round"

    // Top left
    ctx.beginPath(); ctx.moveTo(x + corner, y); ctx.lineTo(x, y); ctx.lineTo(x, y + corner); ctx.stroke()
    // Top right
    ctx.beginPath(); ctx.moveTo(x + w - corner, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + corner); ctx.stroke()
    // Bottom left
    ctx.beginPath(); ctx.moveTo(x + corner, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - corner); ctx.stroke()
    // Bottom right
    ctx.beginPath(); ctx.moveTo(x + w - corner, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - corner); ctx.stroke()

    // Label above box
    ctx.globalAlpha = 1
    ctx.fillStyle   = color
    ctx.font        = "bold 12px monospace"
    ctx.fillText(fatigue ? "⚠ FATIGUE" : "TRACKING", x + 4, y - 8)

  }, [showTracking, faceBox, fatigue, status])

  useEffect(() => {
    if (running && showTracking) {
      trackingRef.current = setInterval(drawTrackingBox, 100)
    }
    return () => clearInterval(trackingRef.current)
  }, [running, showTracking, drawTrackingBox])

  const stopCamera = useCallback(() => {
    isRunningRef.current = false
    clearInterval(intervalRef.current)
    clearInterval(trackingRef.current)
    intervalRef.current = null
    trackingRef.current = null
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    if (overlayRef.current) {
      const ctx = overlayRef.current.getContext("2d")
      ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height)
    }
    setRunning(false)
    setStatus("idle")
    setFatigue(false)
    setFaceDetected(false)
    setEar(0)
    setCnnScore(0)
    closedFramesRef.current = 0
    setClosedFrames(0)
  }, [])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      videoRef.current.srcObject = stream
      isRunningRef.current = true
      setRunning(true)
      setStatus("alert")
    } catch {
      alert("Cannot access camera!")
    }
  }

  const captureAndPredict = useCallback(async () => {
    if (!isRunningRef.current) return
    if (!videoRef.current || !canvasRef.current) return
    if (videoRef.current.readyState !== 4) return

    const canvas = canvasRef.current
    const ctx    = canvas.getContext("2d")
    canvas.width  = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    ctx.drawImage(videoRef.current, 0, 0)

    canvas.toBlob(async (blob) => {
      if (!blob || !isRunningRef.current) return
      const formData = new FormData()
      formData.append("file", blob, "frame.jpg")

      try {
        const res  = await fetch("http://localhost:8000/predict", {
          method: "POST", body: formData
        })
        const data = await res.json()
        if (!isRunningRef.current) return

        setFaceDetected(data.face_detected)
        setFaceBox(data.face_box || null)
        setEar(data.ear ?? 0)
        setCnnScore(data.cnn_score ?? 0)

        if (!data.face_detected) {
          setStatus("noface")
          closedFramesRef.current = 0
          setClosedFrames(0)
          return
        }

        const eyesClosed = data.ear > 0 && data.ear < earThreshold

        if (eyesClosed) {
          closedFramesRef.current += 1
        } else {
          closedFramesRef.current = Math.max(0, closedFramesRef.current - 1)
          if (closedFramesRef.current === 0) setFatigue(false)
        }

        setClosedFrames(closedFramesRef.current)

        if (closedFramesRef.current >= CLOSED_FRAMES_LIMIT) {
          setFatigue(true)
          setStatus("fatigue")
        } else if (eyesClosed) {
          setStatus("closed")
        } else {
          setStatus("alert")
        }

      } catch {
        if (isRunningRef.current) setApiOnline(false)
      }
    }, "image/jpeg", 0.8)
  }, [earThreshold])

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(captureAndPredict, 200)
    }
    return () => clearInterval(intervalRef.current)
  }, [running, captureAndPredict])

  const statusConfig = {
    idle:    { color: "text-gray-400",   bg: "bg-gray-800",   border: "border-gray-600",  label: "System idle",       icon: <Eye size={20}/> },
    alert:   { color: "text-green-400",  bg: "bg-green-900",  border: "border-green-500", label: "Worker alert",      icon: <Eye size={20}/> },
    closed:  { color: "text-yellow-400", bg: "bg-yellow-900", border: "border-yellow-500",label: "Eyes closing...",   icon: <EyeOff size={20}/> },
    fatigue: { color: "text-red-400",    bg: "bg-red-900",    border: "border-red-500",   label: "FATIGUE DETECTED!", icon: <AlertTriangle size={20}/> },
    noface:  { color: "text-orange-400", bg: "bg-orange-900", border: "border-orange-500",label: "No face detected",  icon: <Activity size={20}/> },
  }

  const s = statusConfig[status]

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">
      <div className="max-w-6xl mx-auto">

        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">Worker Fatigue Detection</h1>
            <p className="text-gray-400 mt-1 text-sm">Real-time monitoring using Computer Vision & Deep Learning</p>
          </div>
          <div className={`flex items-center gap-2 text-sm px-3 py-1.5 rounded-full border ${
            apiOnline ? "text-green-400 bg-green-900/30 border-green-700"
                      : "text-red-400 bg-red-900/30 border-red-700"
          }`}>
            {apiOnline ? <Wifi size={14}/> : <WifiOff size={14}/>}
            {apiOnline ? "API Online" : "API Offline"}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">

            <div className={`relative rounded-2xl overflow-hidden border-2 transition-all duration-300 ${
              fatigue             ? "border-red-500 shadow-lg shadow-red-500/20" :
              status === "closed" ? "border-yellow-500" :
              status === "alert"  ? "border-green-500"  : "border-gray-700"
            }`}>
              <video
                ref={videoRef}
                autoPlay playsInline muted
                className="w-full aspect-video object-cover bg-gray-900"
                style={{ transform: "scaleX(-1)" }}
              />
              <canvas
                ref={overlayRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ transform: "scaleX(-1)" }}
              />
              {!running && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                  <p className="text-gray-400 text-lg">Camera off</p>
                </div>
              )}
              {fatigue && (
                <div className="absolute inset-0 border-4 border-red-500 rounded-2xl animate-pulse pointer-events-none"/>
              )}
              {running && (
                <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 px-3 py-1.5 rounded-full text-xs">
                  <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"/>
                  LIVE
                </div>
              )}
              {running && (
                <button
                  onClick={() => setShowTracking(v => !v)}
                  className="absolute top-3 right-3 bg-black/60 hover:bg-black/80 px-3 py-1.5 rounded-full text-xs transition-colors"
                >
                  {showTracking ? "Hide tracking" : "Show tracking"}
                </button>
              )}
              {running && (
                <div className="absolute bottom-3 left-3 bg-black/60 px-3 py-1.5 rounded-full text-xs text-gray-300">
                  EAR: {ear.toFixed(3)}
                </div>
              )}
            </div>

            <canvas ref={canvasRef} className="hidden"/>

            <div className="flex gap-3">
              {!running ? (
                <button
                  onClick={startCamera}
                  disabled={!apiOnline}
                  className="flex-1 py-3 bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-xl font-medium transition-colors"
                >
                  Start monitoring
                </button>
              ) : (
                <button
                  onClick={stopCamera}
                  className="flex-1 py-3 bg-red-600 hover:bg-red-500 rounded-xl font-medium transition-colors"
                >
                  Stop monitoring
                </button>
              )}
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <div className="flex justify-between items-center mb-2">
                <p className="text-xs text-gray-400 uppercase tracking-wider">EAR threshold</p>
                <p className="text-sm font-medium text-white">{earThreshold.toFixed(2)}</p>
              </div>
              <input
                type="range" min="0.10" max="0.30" step="0.01"
                value={earThreshold}
                onChange={e => setEarThreshold(parseFloat(e.target.value))}
                className="w-full accent-green-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0.10 (less sensitive)</span>
                <span>0.30 (more sensitive)</span>
              </div>
            </div>

          </div>

          <div className="space-y-4">
            <div className={`rounded-xl border p-4 transition-all duration-300 ${s.bg}/30 ${s.border}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className={s.color}>{s.icon}</span>
                <span className="text-xs text-gray-400 uppercase tracking-wider">Status</span>
              </div>
              <p className={`text-lg font-semibold ${s.color}`}>{s.label}</p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
                <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">EAR Score</p>
                <p className="text-2xl font-bold text-white">{ear.toFixed(3)}</p>
                <p className="text-xs text-gray-500 mt-1">Threshold: {earThreshold.toFixed(2)}</p>
              </div>
              <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
                <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">CNN Score</p>
                <p className="text-2xl font-bold text-white">{cnnScore.toFixed(3)}</p>
                <p className="text-xs text-gray-500 mt-1">Threshold: 0.5</p>
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <div className="flex justify-between items-center mb-2">
                <p className="text-xs text-gray-400 uppercase tracking-wider">Closed frames</p>
                <p className="text-sm font-medium text-white">{closedFrames} / {CLOSED_FRAMES_LIMIT}</p>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-200 ${
                    fatigue ? "bg-red-500" :
                    closedFrames > CLOSED_FRAMES_LIMIT * 0.6 ? "bg-yellow-500" : "bg-green-500"
                  }`}
                  style={{ width: `${Math.min((closedFrames / CLOSED_FRAMES_LIMIT) * 100, 100)}%` }}
                />
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-3">Detection info</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Face detected</span>
                  <span className={faceDetected ? "text-green-400" : "text-red-400"}>
                    {faceDetected ? "Yes" : "No"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Eyes state</span>
                  <span className={status === "alert" ? "text-green-400" : "text-yellow-400"}>
                    {status === "alert" ? "Open" : status === "closed" || status === "fatigue" ? "Closed" : "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Tracking overlay</span>
                  <span className={showTracking ? "text-green-400" : "text-gray-500"}>
                    {showTracking ? "Visible" : "Hidden"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Alert level</span>
                  <span className={fatigue ? "text-red-400" : "text-green-400"}>
                    {fatigue ? "HIGH" : "Normal"}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
              <p className="text-xs text-gray-400 uppercase tracking-wider mb-3">How it works</p>
              <div className="space-y-2 text-xs text-gray-400">
                <p>1. Webcam captures your face</p>
                <p>2. MediaPipe detects facial landmarks</p>
                <p>3. EAR score measures eye openness</p>
                <p>4. Alert triggers after {CLOSED_FRAMES_LIMIT} closed frames</p>
                <p>5. Tracking box shows face position</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}