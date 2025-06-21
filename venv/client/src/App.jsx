import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [text, setText] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [voice, setVoice] = useState(null);
  const [voices, setVoices] = useState([]);
  const [rate, setRate] = useState(1);
  const [pitch, setPitch] = useState(1);
  const [volume, setVolume] = useState(1);
  const [darkMode, setDarkMode] = useState(false);
  const [audioUrl, setAudioUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatResponse, setChatResponse] = useState("");
  const [chatAudioUrl, setChatAudioUrl] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("llama2");

  const synthRef = useRef(window.speechSynthesis);
  const utteranceRef = useRef(null);

  React.useEffect(() => {
    const loadVoices = () => {
      const availableVoices = synthRef.current.getVoices();
      setVoices(availableVoices);
      if (availableVoices.length > 0 && !voice) {
        setVoice(availableVoices[0]);
      }
    };

    loadVoices();
    synthRef.current.addEventListener("voiceschanged", loadVoices);

    return () => {
      synthRef.current.removeEventListener("voiceschanged", loadVoices);
    };
  }, [voice]);

  // Fetch available models from backend
  React.useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get("http://localhost:5001/api/models");
        if (response.data.success) {
          setAvailableModels(response.data.models);
        }
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    };

    fetchModels();
  }, []);

  const handleAskQuestion = async () => {
    if (!chatQuestion.trim()) return;

    setChatLoading(true);
    setChatResponse("");
    setChatAudioUrl("");
    setError("");

    try {
      const response = await axios.post(
        "http://localhost:5001/api/chat-and-speak",
        {
          question: chatQuestion,
          language: voice?.lang?.split("-")[0] || "en",
          model: selectedModel,
        }
      );

      const data = response.data;

      if (data.success) {
        setChatResponse(data.response);
        if (data.audio_url) {
          setChatAudioUrl(`http://localhost:5001${data.audio_url}`);
        }
      } else {
        setError("Error: No valid response from the AI model.");
      }
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message;
      setError(`Error: ${errorMessage}`);
      console.error("Chat request failed:", err);
    } finally {
      setChatLoading(false);
    }
  };

  const handleSpeak = () => {
    if (!text.trim()) return;

    if (isPlaying) {
      synthRef.current.cancel();
      setIsPlaying(false);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = voice;
    utterance.rate = rate;
    utterance.pitch = pitch;
    utterance.volume = volume;

    utterance.onstart = () => setIsPlaying(true);
    utterance.onend = () => setIsPlaying(false);
    utterance.onerror = () => setIsPlaying(false);

    utteranceRef.current = utterance;
    synthRef.current.speak(utterance);
  };

  const handleDownload = async () => {
    if (!text.trim()) return;

    try {
      // Create a simple audio recording simulation
      // In a real app, you'd use a more sophisticated audio recording method
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.voice = voice;
      utterance.rate = rate;
      utterance.pitch = pitch;
      utterance.volume = volume;

      // For demo purposes, we'll create a simple blob
      // In production, you'd want to use MediaRecorder API or external service
      const blob = new Blob([text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = `speech-${Date.now()}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Download failed:", error);
    }
  };

  const getVoicesByLanguage = () => {
    const grouped = {};
    voices.forEach((voice) => {
      const lang = voice.lang.split("-")[0];
      if (!grouped[lang]) grouped[lang] = [];
      grouped[lang].push(voice);
    });
    return grouped;
  };

  const voicesByLanguage = getVoicesByLanguage();

  // Correct useEffect usage: log audioUrl when it changes
  useEffect(() => {
    if (audioUrl) {
      console.log("Audio URL updated:", audioUrl);
    }
  }, [audioUrl]);

  const handleSendToBackend = async () => {
    setLoading(true);
    setError("");
    setAudioUrl("");
    try {
      const response = await axios.post(
        "http://localhost:5001/api/text-to-speech",
        {
          text: text,
          language: voice?.lang?.split("-")[0] || "en",
        }
      );
      if (response.data && response.data.audio_url) {
        setAudioUrl(`http://localhost:5001${response.data.audio_url}`);
      } else {
        setError("No audio URL returned from backend.");
      }
    } catch (err) {
      setError(err.response?.data?.error || "Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`app ${darkMode ? "dark-mode" : ""}`}>
      <div className="container">
        <header className="header">
          <h1>🎤 Text-to-Voice Translator</h1>
          <button
            className="theme-toggle"
            onClick={() => setDarkMode(!darkMode)}
            aria-label="Toggle theme"
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </header>

        <main className="main">
          <div className="text-section">
            <label htmlFor="text-input" className="label">
              Enter your text:
            </label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type or paste your text here..."
              className="text-input"
              rows={6}
            />
            <button
              onClick={handleSendToBackend}
              disabled={!text.trim() || loading}
              className="btn btn-secondary"
              style={{ marginTop: 16 }}
            >
              {loading ? "Converting..." : "Convert with Flask Backend"}
            </button>
            {audioUrl && (
              <div style={{ marginTop: 16 }}>
                <audio controls src={audioUrl} />
                <a href={audioUrl} download style={{ marginLeft: 12 }}>
                  Download MP3
                </a>
              </div>
            )}
            {error && (
              <div style={{ color: "#ff6b6b", marginTop: 12 }}>{error}</div>
            )}
          </div>

          <div className="chat-section" style={{ marginTop: 32 }}>
            <h2>🧠 Ask a Question (Ollama + TTS)</h2>

            <div style={{ marginBottom: 12 }}>
              <label htmlFor="model-select" className="label">
                AI Model:
              </label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="select"
                style={{ marginLeft: 8 }}
              >
                {availableModels.length > 0 ? (
                  availableModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))
                ) : (
                  <option value="llama2">llama2</option>
                )}
              </select>
            </div>

            <textarea
              placeholder="Ask anything (e.g., What is AI?)"
              value={chatQuestion}
              onChange={(e) => setChatQuestion(e.target.value)}
              rows={4}
              className="text-input"
              style={{ marginBottom: 12 }}
            />
            <button
              onClick={handleAskQuestion}
              disabled={!chatQuestion.trim() || chatLoading}
              className="btn btn-primary"
            >
              {chatLoading ? "Thinking..." : "Ask & Get Audio"}
            </button>

            {chatResponse && (
              <div style={{ marginTop: 16 }}>
                <strong>Answer:</strong>
                <p>{chatResponse}</p>
              </div>
            )}

            {error && (
              <div style={{ color: "#ff6b6b", marginTop: 12 }}>{error}</div>
            )}

            {chatAudioUrl && (
              <div style={{ marginTop: 12 }}>
                <audio controls src={chatAudioUrl} />
                <a href={chatAudioUrl} download style={{ marginLeft: 12 }}>
                  Download Audio
                </a>
              </div>
            )}
          </div>

          <div className="controls-section">
            <div className="voice-controls">
              <div className="control-group">
                <label htmlFor="voice-select" className="label">
                  Voice:
                </label>
                <select
                  id="voice-select"
                  value={voice?.name || ""}
                  onChange={(e) => {
                    const selectedVoice = voices.find(
                      (v) => v.name === e.target.value
                    );
                    setVoice(selectedVoice);
                  }}
                  className="select"
                >
                  {Object.entries(voicesByLanguage).map(
                    ([lang, langVoices]) => (
                      <optgroup key={lang} label={lang.toUpperCase()}>
                        {langVoices.map((voice) => (
                          <option key={voice.name} value={voice.name}>
                            {voice.name} ({voice.lang})
                          </option>
                        ))}
                      </optgroup>
                    )
                  )}
                </select>
              </div>

              <div className="control-group">
                <label htmlFor="rate-slider" className="label">
                  Speed: {rate}x
                </label>
                <input
                  id="rate-slider"
                  type="range"
                  min="0.1"
                  max="3"
                  step="0.1"
                  value={rate}
                  onChange={(e) => setRate(parseFloat(e.target.value))}
                  className="slider"
                />
              </div>

              <div className="control-group">
                <label htmlFor="pitch-slider" className="label">
                  Pitch: {pitch}
                </label>
                <input
                  id="pitch-slider"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={pitch}
                  onChange={(e) => setPitch(parseFloat(e.target.value))}
                  className="slider"
                />
              </div>

              <div className="control-group">
                <label htmlFor="volume-slider" className="label">
                  Volume: {Math.round(volume * 100)}%
                </label>
                <input
                  id="volume-slider"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={volume}
                  onChange={(e) => setVolume(parseFloat(e.target.value))}
                  className="slider"
                />
              </div>
            </div>

            <div className="action-buttons">
              <button
                onClick={handleSpeak}
                disabled={!text.trim()}
                className={`btn btn-primary ${isPlaying ? "playing" : ""}`}
              >
                {isPlaying ? "⏹️ Stop" : "▶️ Play"}
              </button>

              <button
                onClick={handleDownload}
                disabled={!text.trim()}
                className="btn btn-secondary"
              >
                💾 Download
              </button>
            </div>
          </div>

          <div className="info-section">
            <div className="stats">
              <span>Characters: {text.length}</span>
              <span>
                Words: {text.trim() ? text.trim().split(/\s+/).length : 0}
              </span>
              {voice && <span>Language: {voice.lang}</span>}
            </div>
          </div>
        </main>

        <footer className="footer">
          <p>Built with Web Speech API • Works on all modern browsers</p>
        </footer>
      </div>
    </div>
  );
}
