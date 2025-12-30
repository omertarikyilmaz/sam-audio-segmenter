/**
 * SAMAudioInterface - Audio Source Separation using Meta's SAM-Audio
 * 
 * Features:
 * - Upload audio files up to 65 minutes
 * - Custom text prompts (default: "A news anchor speaking")
 * - Three-track waveform visualization with Canvas
 * - Individual and synchronized audio playback
 * - Download separated audio tracks
 */
import { useState, useRef, useEffect, useCallback } from 'react'
import { Upload, Loader2, AlertCircle, CheckCircle, Download, Play, Pause, Volume2, VolumeX, Music, Square } from 'lucide-react'

const DEFAULT_PROMPT = "Human speech, people talking, voices, conversation, and spoken words"

// WaveformVisualizer Component - Canvas-based waveform rendering
function WaveformVisualizer({ analyserNode, isPlaying, color, height = 60 }) {
    const canvasRef = useRef(null)
    const animationRef = useRef(null)

    useEffect(() => {
        if (!analyserNode || !canvasRef.current) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const bufferLength = analyserNode.frequencyBinCount
        const dataArray = new Uint8Array(bufferLength)

        const draw = () => {
            if (!isPlaying) {
                // Draw static waveform when not playing
                ctx.fillStyle = 'rgba(30, 30, 40, 0.5)'
                ctx.fillRect(0, 0, canvas.width, canvas.height)

                ctx.strokeStyle = color + '40'
                ctx.lineWidth = 2
                ctx.beginPath()
                ctx.moveTo(0, canvas.height / 2)
                ctx.lineTo(canvas.width, canvas.height / 2)
                ctx.stroke()
                return
            }

            animationRef.current = requestAnimationFrame(draw)
            analyserNode.getByteTimeDomainData(dataArray)

            ctx.fillStyle = 'rgba(30, 30, 40, 0.3)'
            ctx.fillRect(0, 0, canvas.width, canvas.height)

            ctx.lineWidth = 2
            ctx.strokeStyle = color
            ctx.beginPath()

            const sliceWidth = canvas.width / bufferLength
            let x = 0

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0
                const y = (v * canvas.height) / 2

                if (i === 0) {
                    ctx.moveTo(x, y)
                } else {
                    ctx.lineTo(x, y)
                }
                x += sliceWidth
            }

            ctx.lineTo(canvas.width, canvas.height / 2)
            ctx.stroke()

            // Draw glow effect
            ctx.shadowBlur = 10
            ctx.shadowColor = color
            ctx.stroke()
            ctx.shadowBlur = 0
        }

        draw()

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current)
            }
        }
    }, [analyserNode, isPlaying, color])

    return (
        <canvas
            ref={canvasRef}
            width={400}
            height={height}
            style={{
                width: '100%',
                height: `${height}px`,
                borderRadius: '8px',
                background: 'rgba(30, 30, 40, 0.5)'
            }}
        />
    )
}

// FrequencyVisualizer Component - Frequency spectrum bars
function FrequencyVisualizer({ analyserNode, isPlaying, color, height = 60 }) {
    const canvasRef = useRef(null)
    const animationRef = useRef(null)

    useEffect(() => {
        if (!analyserNode || !canvasRef.current) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const bufferLength = analyserNode.frequencyBinCount
        const dataArray = new Uint8Array(bufferLength)

        const draw = () => {
            if (!isPlaying) {
                // Draw static bars when not playing
                ctx.fillStyle = 'rgba(30, 30, 40, 0.5)'
                ctx.fillRect(0, 0, canvas.width, canvas.height)

                const barCount = 32
                const barWidth = canvas.width / barCount - 2

                for (let i = 0; i < barCount; i++) {
                    const barHeight = Math.random() * 10 + 5
                    const x = i * (barWidth + 2)
                    ctx.fillStyle = color + '30'
                    ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight)
                }
                return
            }

            animationRef.current = requestAnimationFrame(draw)
            analyserNode.getByteFrequencyData(dataArray)

            ctx.fillStyle = 'rgba(30, 30, 40, 0.3)'
            ctx.fillRect(0, 0, canvas.width, canvas.height)

            const barCount = 32
            const barWidth = canvas.width / barCount - 2

            for (let i = 0; i < barCount; i++) {
                const dataIndex = Math.floor(i * bufferLength / barCount)
                const barHeight = (dataArray[dataIndex] / 255) * canvas.height * 0.9

                const x = i * (barWidth + 2)

                // Gradient color based on height
                const gradient = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - barHeight)
                gradient.addColorStop(0, color)
                gradient.addColorStop(1, color + '80')

                ctx.fillStyle = gradient
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight)

                // Glow effect
                ctx.shadowBlur = 5
                ctx.shadowColor = color
                ctx.fillRect(x, canvas.height - barHeight, barWidth, 2)
                ctx.shadowBlur = 0
            }
        }

        draw()

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current)
            }
        }
    }, [analyserNode, isPlaying, color])

    return (
        <canvas
            ref={canvasRef}
            width={400}
            height={height}
            style={{
                width: '100%',
                height: `${height}px`,
                borderRadius: '8px',
                background: 'rgba(30, 30, 40, 0.5)'
            }}
        />
    )
}

// AudioTrackPlayer Component - Individual track player with visualization
function AudioTrackPlayer({ trackKey, label, color, downloadUrl, taskId, isGlobalPlaying, globalTime, onTimeUpdate }) {
    const audioRef = useRef(null)
    const audioContextRef = useRef(null)
    const analyserRef = useRef(null)
    const sourceRef = useRef(null)

    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)
    const [volume, setVolume] = useState(1)
    const [isMuted, setIsMuted] = useState(false)
    const [isLoaded, setIsLoaded] = useState(false)

    // Initialize Web Audio API
    useEffect(() => {
        if (!audioRef.current || !downloadUrl) return

        const audio = audioRef.current

        const initAudioContext = () => {
            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
                analyserRef.current = audioContextRef.current.createAnalyser()
                analyserRef.current.fftSize = 256

                sourceRef.current = audioContextRef.current.createMediaElementSource(audio)
                sourceRef.current.connect(analyserRef.current)
                analyserRef.current.connect(audioContextRef.current.destination)
            }
        }

        audio.addEventListener('canplaythrough', () => {
            setIsLoaded(true)
            initAudioContext()
        })

        audio.addEventListener('loadedmetadata', () => {
            setDuration(audio.duration)
        })

        audio.addEventListener('timeupdate', () => {
            setCurrentTime(audio.currentTime)
            if (onTimeUpdate) onTimeUpdate(audio.currentTime)
        })

        audio.addEventListener('ended', () => {
            setIsPlaying(false)
        })

        // Try to initiate audio context on first user interaction
        const handleInteraction = () => {
            initAudioContext()
            if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
                audioContextRef.current.resume()
            }
        }

        document.addEventListener('click', handleInteraction, { once: true })

        return () => {
            document.removeEventListener('click', handleInteraction)
        }
    }, [downloadUrl, onTimeUpdate])

    // Sync with global playback
    useEffect(() => {
        if (audioRef.current && typeof globalTime === 'number') {
            const diff = Math.abs(audioRef.current.currentTime - globalTime)
            if (diff > 0.5) {
                audioRef.current.currentTime = globalTime
            }
        }
    }, [globalTime])

    useEffect(() => {
        if (!audioRef.current) return

        if (isGlobalPlaying) {
            audioRef.current.play().catch(() => { })
            setIsPlaying(true)
        } else {
            audioRef.current.pause()
            setIsPlaying(false)
        }
    }, [isGlobalPlaying])

    const togglePlay = () => {
        if (!audioRef.current || !isLoaded) return

        if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
            audioContextRef.current.resume()
        }

        if (isPlaying) {
            audioRef.current.pause()
            setIsPlaying(false)
        } else {
            audioRef.current.play().catch(console.error)
            setIsPlaying(true)
        }
    }

    const handleSeek = (e) => {
        if (!audioRef.current || !duration) return
        const rect = e.currentTarget.getBoundingClientRect()
        const percent = (e.clientX - rect.left) / rect.width
        const time = Math.max(0, Math.min(percent * duration, duration))
        audioRef.current.currentTime = time
        setCurrentTime(time)
    }

    const toggleMute = () => {
        if (audioRef.current) {
            audioRef.current.muted = !isMuted
            setIsMuted(!isMuted)
        }
    }

    const handleVolumeChange = (e) => {
        const newVolume = parseFloat(e.target.value)
        setVolume(newVolume)
        if (audioRef.current) {
            audioRef.current.volume = newVolume
        }
    }

    const formatTime = (seconds) => {
        if (!seconds || !isFinite(seconds)) return '0:00'
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    const downloadTrack = () => {
        if (taskId) {
            window.open(`/api/v1/pipelines/sam-audio/download/${taskId}/${trackKey}`, '_blank')
        }
    }

    return (
        <div style={{
            background: `linear-gradient(135deg, ${color}10, ${color}05)`,
            border: `1px solid ${color}30`,
            borderRadius: '12px',
            padding: '1rem',
            marginBottom: '1rem'
        }}>
            {/* Track Header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '0.75rem'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '50%',
                        background: color,
                        boxShadow: `0 0 10px ${color}`
                    }} />
                    <span style={{ fontWeight: 600, color: color, fontSize: '1rem' }}>
                        {label}
                    </span>
                </div>

                <button
                    onClick={downloadTrack}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.4rem 0.75rem',
                        background: color,
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '0.8rem',
                        fontWeight: 500
                    }}
                >
                    <Download size={14} />
                    İndir
                </button>
            </div>

            {/* Waveform Visualization */}
            <div style={{ marginBottom: '0.75rem' }}>
                <WaveformVisualizer
                    analyserNode={analyserRef.current}
                    isPlaying={isPlaying}
                    color={color}
                    height={50}
                />
            </div>

            {/* Frequency Visualization */}
            <div style={{ marginBottom: '0.75rem' }}>
                <FrequencyVisualizer
                    analyserNode={analyserRef.current}
                    isPlaying={isPlaying}
                    color={color}
                    height={40}
                />
            </div>

            {/* Controls */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem'
            }}>
                {/* Play/Pause Button */}
                <button
                    onClick={togglePlay}
                    disabled={!isLoaded}
                    style={{
                        width: '36px',
                        height: '36px',
                        borderRadius: '50%',
                        background: isLoaded ? color : 'var(--border-color)',
                        border: 'none',
                        cursor: isLoaded ? 'pointer' : 'not-allowed',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        opacity: isLoaded ? 1 : 0.5
                    }}
                >
                    {isPlaying ? <Pause size={16} color="white" /> : <Play size={16} color="white" />}
                </button>

                {/* Progress Bar */}
                <div style={{ flex: 1 }}>
                    <div
                        onClick={handleSeek}
                        style={{
                            height: '6px',
                            background: 'var(--border-color)',
                            borderRadius: '3px',
                            cursor: 'pointer',
                            position: 'relative'
                        }}
                    >
                        <div style={{
                            width: `${duration ? (currentTime / duration) * 100 : 0}%`,
                            height: '100%',
                            background: color,
                            borderRadius: '3px',
                            transition: 'width 0.1s ease'
                        }} />
                    </div>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.75rem',
                        color: 'var(--text-secondary)',
                        marginTop: '0.25rem'
                    }}>
                        <span>{formatTime(currentTime)}</span>
                        <span>{formatTime(duration)}</span>
                    </div>
                </div>

                {/* Volume Control */}
                <button
                    onClick={toggleMute}
                    style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '0.25rem'
                    }}
                >
                    {isMuted ? <VolumeX size={18} color={color} /> : <Volume2 size={18} color={color} />}
                </button>

                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={volume}
                    onChange={handleVolumeChange}
                    style={{
                        width: '60px',
                        accentColor: color
                    }}
                />
            </div>

            {/* Hidden Audio Element */}
            <audio
                ref={audioRef}
                src={downloadUrl}
                preload="metadata"
                crossOrigin="anonymous"
            />
        </div>
    )
}

export default function SAMAudioInterface() {
    const [audioFile, setAudioFile] = useState(null)
    const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
    const [taskId, setTaskId] = useState(null)
    const [status, setStatus] = useState(null)
    const [error, setError] = useState(null)
    const [loading, setLoading] = useState(false)
    const fileInputRef = useRef(null)

    // Global playback state
    const [isGlobalPlaying, setIsGlobalPlaying] = useState(false)
    const [globalTime, setGlobalTime] = useState(0)

    const handleFileChange = (e) => {
        const file = e.target.files[0]
        if (file) {
            if (file.size > 500 * 1024 * 1024) {
                setError('Dosya çok büyük. Maksimum 500MB.')
                return
            }
            setAudioFile(file)
            setStatus(null)
            setError(null)
            setTaskId(null)
        }
    }

    const handleSubmit = async () => {
        if (!audioFile) {
            setError('Lütfen bir ses dosyası seçin')
            return
        }

        setLoading(true)
        setError(null)
        setStatus(null)
        setIsGlobalPlaying(false)

        const formData = new FormData()
        formData.append('file', audioFile)
        formData.append('prompt', prompt || DEFAULT_PROMPT)

        try {
            const response = await fetch('/api/v1/pipelines/sam-audio/separate', {
                method: 'POST',
                body: formData
            })

            if (!response.ok) {
                const err = await response.json().catch(() => ({}))
                throw new Error(err.detail || `HTTP ${response.status}`)
            }

            const data = await response.json()
            setTaskId(data.task_id)
            setStatus({ status: 'pending', progress: 0, message: 'Task started...' })
        } catch (err) {
            setError(err.message)
            setLoading(false)
        }
    }

    // Poll for status updates
    useEffect(() => {
        if (!taskId) return

        const pollStatus = async () => {
            try {
                const response = await fetch(`/api/v1/pipelines/sam-audio/status/${taskId}`)
                const data = await response.json()
                setStatus(data)

                if (data.status === 'completed' || data.status === 'failed') {
                    setLoading(false)
                    if (data.status === 'failed') {
                        setError(data.error || data.message)
                    }
                }
            } catch (err) {
                console.error('Status poll error:', err)
            }
        }

        const interval = setInterval(pollStatus, 2000)
        pollStatus()

        return () => clearInterval(interval)
    }, [taskId])

    const toggleGlobalPlay = () => {
        setIsGlobalPlaying(!isGlobalPlaying)
    }

    const stopAll = () => {
        setIsGlobalPlaying(false)
        setGlobalTime(0)
    }

    const tracks = [
        { key: 'original', label: 'Orijinal Ses', color: '#06b6d4' },
        { key: 'isolated', label: 'İnsan Sesleri (Haber İçeriği)', color: '#10b981' },
        { key: 'residual', label: 'Arka Plan (Müzik/Efekt)', color: '#ec4899' },
        { key: 'cleaned', label: 'Temizlenmiş Haber', color: '#8b5cf6' }
    ]

    return (
        <div className="animate-fade-in" style={{ maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ background: 'rgba(168, 85, 247, 0.1)', padding: '0.75rem', borderRadius: '0.75rem' }}>
                    <Music size={32} color="#a855f7" />
                </div>
                <div>
                    <h2 style={{ fontSize: '1.75rem', fontWeight: 700 }}>SAM-Audio Source Separation</h2>
                    <p style={{ color: 'var(--text-secondary)' }}>Meta'nın SAM-Audio modeli ile ses kaynağı ayırma</p>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: '2rem' }}>
                {/* Left Column - Upload & Settings */}
                <div>
                    {/* Prompt Input */}
                    <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600, color: '#a855f7' }}>
                            Text Prompt (İzole edilecek ses)
                        </label>
                        <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="A news anchor speaking"
                            style={{
                                width: '100%',
                                padding: '0.75rem',
                                background: 'var(--bg-color)',
                                border: '1px solid var(--border-color)',
                                borderRadius: '0.5rem',
                                color: 'var(--text-primary)',
                                fontSize: '0.9rem'
                            }}
                        />
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                            Örnek: "A man speaking", "Background music", "Dog barking"
                        </p>
                    </div>

                    {/* File Upload */}
                    <div
                        className="glass-panel"
                        style={{ padding: '2rem', textAlign: 'center', cursor: 'pointer', marginBottom: '1.5rem' }}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            accept=".wav,.mp3,.ogg,.flac,.m4a,.aac"
                            style={{ display: 'none' }}
                        />
                        <Upload size={48} color="#a855f7" style={{ margin: '0 auto 1rem' }} />
                        <p style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                            {audioFile ? audioFile.name : 'Ses Dosyası Yükle'}
                        </p>
                        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            WAV, MP3, OGG, FLAC (Max 65 dakika)
                        </p>
                    </div>

                    {/* Submit Button */}
                    <button
                        className="btn btn-primary"
                        style={{ width: '100%', background: '#a855f7', borderColor: '#a855f7' }}
                        onClick={handleSubmit}
                        disabled={!audioFile || loading}
                    >
                        {loading ? (
                            <>
                                <Loader2 className="loading-spinner" size={18} />
                                İşleniyor...
                            </>
                        ) : (
                            <>
                                <Music size={18} />
                                Ses Ayırma Başlat
                            </>
                        )}
                    </button>

                    {/* Progress */}
                    {status && status.status === 'processing' && (
                        <div className="glass-panel" style={{ padding: '1.5rem', marginTop: '1.5rem' }}>
                            <div style={{ marginBottom: '1rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                    <span style={{ fontWeight: 600, color: '#a855f7' }}>
                                        {status.progress}%
                                    </span>
                                    <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                                        {status.message}
                                    </span>
                                </div>
                                <div style={{
                                    width: '100%',
                                    height: '8px',
                                    background: 'var(--surface-color)',
                                    borderRadius: '4px',
                                    overflow: 'hidden'
                                }}>
                                    <div style={{
                                        width: `${status.progress}%`,
                                        height: '100%',
                                        background: 'linear-gradient(90deg, #a855f7, #c084fc)',
                                        transition: 'width 0.3s ease'
                                    }} />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Error */}
                    {error && (
                        <div style={{
                            marginTop: '1.5rem',
                            color: '#ef4444',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            padding: '1rem',
                            background: 'rgba(239, 68, 68, 0.1)',
                            borderRadius: '0.5rem'
                        }}>
                            <AlertCircle size={20} />
                            {error}
                        </div>
                    )}
                </div>

                {/* Right Column - Audio Player & Visualizations */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '1.5rem'
                    }}>
                        <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                            Ses Önizleme & Görselleştirme
                        </h3>

                        {status?.status === 'completed' && (
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                                <button
                                    onClick={toggleGlobalPlay}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.5rem',
                                        padding: '0.5rem 1rem',
                                        background: isGlobalPlaying ? '#ef4444' : '#10b981',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        fontWeight: 500
                                    }}
                                >
                                    {isGlobalPlaying ? <Pause size={16} /> : <Play size={16} />}
                                    {isGlobalPlaying ? 'Duraklat' : 'Tümünü Oynat'}
                                </button>
                                <button
                                    onClick={stopAll}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.5rem',
                                        padding: '0.5rem 1rem',
                                        background: 'var(--surface-color)',
                                        color: 'var(--text-primary)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: '6px',
                                        cursor: 'pointer',
                                        fontWeight: 500
                                    }}
                                >
                                    <Square size={16} />
                                    Durdur
                                </button>
                            </div>
                        )}
                    </div>

                    {status?.status === 'completed' ? (
                        <>
                            {/* Individual Track Players */}
                            {tracks.map(track => (
                                <AudioTrackPlayer
                                    key={track.key}
                                    trackKey={track.key}
                                    label={track.label}
                                    color={track.color}
                                    downloadUrl={status.downloads?.[track.key]}
                                    taskId={taskId}
                                    isGlobalPlaying={isGlobalPlaying}
                                    globalTime={globalTime}
                                    onTimeUpdate={track.key === 'original' ? setGlobalTime : undefined}
                                />
                            ))}

                            {/* Success Message */}
                            <div style={{
                                marginTop: '1rem',
                                padding: '1rem',
                                background: 'rgba(16, 185, 129, 0.1)',
                                borderRadius: '0.5rem',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem'
                            }}>
                                <CheckCircle size={20} color="#10b981" />
                                <span style={{ color: '#10b981', fontWeight: 500 }}>
                                    Ses ayırma tamamlandı! Prompt: "{status.prompt}"
                                </span>
                            </div>
                        </>
                    ) : (
                        <div style={{
                            textAlign: 'center',
                            padding: '4rem 2rem',
                            color: 'var(--text-secondary)'
                        }}>
                            <Music size={64} style={{ opacity: 0.3, margin: '0 auto 1rem' }} />
                            <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                                Henüz işlem yok
                            </p>
                            <p style={{ fontSize: '0.9rem' }}>
                                Ses dosyası yükleyin ve işlemi başlatın
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
