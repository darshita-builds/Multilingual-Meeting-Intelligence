import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

/**
 * Browser meeting recorder. Feeds the SAME upload endpoint (`api.uploadJob`,
 * i.e. `POST /jobs`) an uploaded file already uses -- no new backend path.
 * The produced file is WebM/Opus (or Ogg/Opus as a fallback), both of which
 * `ingestion.py` already accepts by extension, content-type and magic bytes.
 *
 * State machine: idle -> recording <-> paused -> stopped (preview) -> submitting.
 * The elapsed timer only accumulates while actually recording -- pausing
 * freezes it rather than letting a naive Date.now() diff keep ticking.
 */

const CANDIDATE_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/ogg',
]

function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return null
  return CANDIDATE_MIME_TYPES.find((t) => MediaRecorder.isTypeSupported(t)) || null
}

function extensionFor(mimeType) {
  if (mimeType.includes('ogg')) return 'ogg'
  return 'webm'
}

function formatElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

export default function MeetingRecorder({ onUploaded, onError }) {
  const [phase, setPhase] = useState('idle') // idle | recording | paused | stopped | submitting
  const [elapsedMs, setElapsedMs] = useState(0)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [title, setTitle] = useState('')
  const [languageHint, setLanguageHint] = useState('')
  const [localError, setLocalError] = useState('')

  const streamRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const blobRef = useRef(null)
  const mimeTypeRef = useRef(null)
  const segmentStartRef = useRef(0)
  const accumulatedMsRef = useRef(0)
  const tickRef = useRef(null)

  useEffect(() => {
    return () => {
      // Unmounting mid-recording must not leak the microphone or the object URL.
      stopTimer()
      streamRef.current?.getTracks().forEach((t) => t.stop())
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function startTimer() {
    segmentStartRef.current = Date.now()
    stopTimer()
    tickRef.current = setInterval(() => {
      setElapsedMs(accumulatedMsRef.current + (Date.now() - segmentStartRef.current))
    }, 250)
  }

  function stopTimer() {
    if (tickRef.current) {
      clearInterval(tickRef.current)
      tickRef.current = null
    }
  }

  async function handleStart() {
    setLocalError('')
    const mimeType = pickMimeType()
    if (!mimeType) {
      setLocalError(
        "This browser can't record audio in a format the server accepts. Try Chrome, Edge or Firefox, or upload a file instead."
      )
      return
    }
    mimeTypeRef.current = mimeType

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (err) {
      if (err.name === 'NotAllowedError') {
        setLocalError('Microphone access was denied. Allow microphone access and try again.')
      } else if (err.name === 'NotFoundError') {
        setLocalError('No microphone was found on this device.')
      } else {
        setLocalError(`Could not access the microphone: ${err.message}`)
      }
      return
    }

    streamRef.current = stream
    chunksRef.current = []
    accumulatedMsRef.current = 0
    setElapsedMs(0)

    const recorder = new MediaRecorder(stream, { mimeType })
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
    }
    recorder.start()
    recorderRef.current = recorder
    startTimer()
    setPhase('recording')
  }

  function handlePause() {
    recorderRef.current?.pause()
    accumulatedMsRef.current += Date.now() - segmentStartRef.current
    stopTimer()
    setElapsedMs(accumulatedMsRef.current)
    setPhase('paused')
  }

  function handleResume() {
    recorderRef.current?.resume()
    startTimer()
    setPhase('recording')
  }

  function handleStop() {
    const recorder = recorderRef.current
    if (!recorder) return
    if (phase === 'recording') {
      accumulatedMsRef.current += Date.now() - segmentStartRef.current
    }
    stopTimer()
    setElapsedMs(accumulatedMsRef.current)

    recorder.onstop = () => {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      const blob = new Blob(chunksRef.current, { type: mimeTypeRef.current })
      blobRef.current = blob
      setPreviewUrl(URL.createObjectURL(blob))
      setPhase('stopped')
    }
    recorder.stop()
  }

  function handleDiscard() {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    blobRef.current = null
    chunksRef.current = []
    accumulatedMsRef.current = 0
    setElapsedMs(0)
    setLocalError('')
    setPhase('idle')
  }

  async function handleSubmit() {
    const blob = blobRef.current
    if (!blob || blob.size === 0) {
      setLocalError('Recording is empty. Start again.')
      return
    }
    setPhase('submitting')
    try {
      const ext = extensionFor(mimeTypeRef.current)
      const file = new File([blob], `recording-${Date.now()}.${ext}`, { type: blob.type })
      const job = await api.uploadJob(file, title, languageHint)
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(null)
      blobRef.current = null
      chunksRef.current = []
      accumulatedMsRef.current = 0
      setElapsedMs(0)
      setTitle('')
      setPhase('idle')
      onUploaded(job)
    } catch (err) {
      setPhase('stopped')
      onError(err.message)
    }
  }

  const isEmpty = phase === 'stopped' && (!blobRef.current || blobRef.current.size === 0)

  return (
    <div className="card recorder">
      <h2>Record a meeting</h2>
      <p className="hint">
        Record directly in the browser, then submit it for processing through the same pipeline
        as an uploaded file. This does not produce a live transcript while recording.
      </p>

      {localError && <div className="banner error">{localError}</div>}

      <div className="row" style={{ alignItems: 'center', gap: 12 }}>
        <span
          className={`recorder-dot ${phase === 'recording' ? 'live' : ''}`}
          aria-hidden="true"
        />
        <span className="recorder-timer mono">{formatElapsed(elapsedMs)}</span>
        {phase === 'recording' && <span className="muted">Recording…</span>}
        {phase === 'paused' && <span className="muted">Paused</span>}
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        {phase === 'idle' && (
          <button className="primary" onClick={handleStart}>
            Start recording
          </button>
        )}
        {phase === 'recording' && (
          <>
            <button onClick={handlePause}>Pause</button>
            <button className="danger" onClick={handleStop}>
              Stop
            </button>
          </>
        )}
        {phase === 'paused' && (
          <>
            <button className="primary" onClick={handleResume}>
              Resume
            </button>
            <button className="danger" onClick={handleStop}>
              Stop
            </button>
          </>
        )}
      </div>

      {phase === 'stopped' && (
        <div className="stack" style={{ marginTop: 14 }}>
          <div>
            <label>Preview</label>
            <audio controls src={previewUrl} style={{ width: '100%' }} />
          </div>
          {isEmpty && <div className="banner warn">Recording is empty — start again.</div>}
          <div>
            <label>Meeting title</label>
            <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Sprint review" />
          </div>
          <div>
            <label>Primary language (optional hint)</label>
            <select value={languageHint} onChange={(e) => setLanguageHint(e.target.value)}>
              <option value="">Auto-detect</option>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="mr">Marathi</option>
              <option value="ta">Tamil</option>
            </select>
          </div>
          <div className="row">
            <button className="primary" disabled={isEmpty} onClick={handleSubmit}>
              Process meeting
            </button>
            <button onClick={handleDiscard}>Discard &amp; re-record</button>
          </div>
        </div>
      )}

      {phase === 'submitting' && <p className="muted">Uploading…</p>}
    </div>
  )
}
