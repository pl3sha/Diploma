import { useState, useEffect } from "react";
import axios from "axios";

const MODEL_OPTIONS = [
  {
    value: "base",
    label: "Базовая SD v1.5",
    description: "Стандартная Stable Diffusion без адаптации",
  },
  {
    value: "public",
    label: "Публичная LoRA",
    description: "Pixel Art LoRA (CivitAI)",
  },
  {
    value: "custom",
    label: "Обученная LoRA",
    description: "LoRA, дообученная на собственном датасете",
  },
];

function App() {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState(
    "realistic, 3d, blurry, photographic, smooth, noise, low quality, ugly, deformed"
  );
  const [modelType, setModelType] = useState("base");
  const [outputSize, setOutputSize] = useState(128);
  const [steps, setSteps] = useState(25);
  const [image, setImage] = useState(null);
  const [generatedPrompt, setGeneratedPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await axios.get("/history");
      setHistory(res.data.history || []);
    } catch {
    }
  };

  const generate = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setImage(null);
    setGeneratedPrompt("");

    try {
      const res = await axios.post("/generate", {
        prompt,
        negative_prompt: negativePrompt,
        model_type: modelType,
        output_size: outputSize,
        steps,
      });
      setImage(res.data.image);
      setGeneratedPrompt(res.data.prompt);
      await fetchHistory();
    } catch (e) {
      const detail = e.response?.data?.detail;
      if (detail?.includes("не найдена")) {
        setError(`LoRA-файл не найден. ${detail}`);
      } else {
        setError(detail || "Ошибка генерации. Проверь что бэкенд запущен.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && e.ctrlKey) generate();
  };

  const download = () => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${image}`;
    link.download = `character_${modelType}_${outputSize}px.png`;
    link.click();
  };

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">8-bit Character Generator</h1>
        <p className="subtitle">
          Введи описание персонажа — получи пиксельного героя
        </p>
      </header>

      <div className="main-layout">
        <aside className="panel panel-settings">
          <h2 className="panel-title">Настройки</h2>

          <label className="field-label">Модель</label>
          <div className="model-list">
            {MODEL_OPTIONS.map((m) => (
              <button
                key={m.value}
                className={`model-btn ${modelType === m.value ? "active" : ""}`}
                onClick={() => setModelType(m.value)}
              >
                <span className="model-name">{m.label}</span>
                <span className="model-desc">{m.description}</span>
              </button>
            ))}
          </div>

          <label className="field-label">Размер спрайта</label>
          <div className="size-row">
            {[80, 128].map((s) => (
              <button
                key={s}
                className={`size-btn ${outputSize === s ? "active" : ""}`}
                onClick={() => setOutputSize(s)}
              >
                {s}×{s}
              </button>
            ))}
          </div>

          <label className="field-label">
            Шагов генерации: <strong>{steps}</strong>
          </label>
          <input
            type="range"
            min={10}
            max={50}
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
            className="slider"
          />
          <div className="slider-hints">
            <span>10 (быстро)</span>
            <span>50 (качественно)</span>
          </div>
        </aside>

        <main className="panel panel-main">
          <label className="field-label">Описание персонажа</label>
          <textarea
            className="textarea"
            placeholder="Например: враг-робот с красной бронёй и молотом"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
          />
          <span className="hint">Ctrl+Enter — генерировать</span>

          <label className="field-label" style={{ marginTop: 12 }}>
            Negative prompt
          </label>
          <textarea
            className="textarea textarea-small"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            rows={2}
          />

          <button
            className={`btn-generate ${loading ? "disabled" : ""}`}
            onClick={generate}
            disabled={loading}
          >
            {loading ? (
              <span className="spinner-row">
                <span className="spinner" /> Генерация...
              </span>
            ) : (
              "Сгенерировать"
            )}
          </button>

          {error && <div className="error-box">{error}</div>}

          {image && (
            <div className="result">
              <img
                src={`data:image/png;base64,${image}`}
                alt="Сгенерированный персонаж"
                className="result-image"
              />
              <div className="result-meta">
                <span className="tag">{modelType}</span>
                <span className="tag">{outputSize}×{outputSize} px</span>
              </div>
              {generatedPrompt && (
                <p className="full-prompt">
                  <strong>Полный промпт:</strong> {generatedPrompt}
                </p>
              )}
              <button className="btn-download" onClick={download}>
                ⬇ Скачать PNG
              </button>
            </div>
          )}
        </main>

        <aside className="panel panel-history">
          <div className="history-header">
            <h2 className="panel-title">История</h2>
            <button
              className="btn-refresh"
              onClick={fetchHistory}
              title="Обновить"
            >
              ↻
            </button>
          </div>

          {history.length === 0 ? (
            <p className="empty-state">Пока нет генераций</p>
          ) : (
            <div className="history-grid">
              {history.map((item) => (
                <div key={item.filename} className="history-item">
                  <img
                    src={`data:image/png;base64,${item.image}`}
                    alt={item.filename}
                    className="history-image"
                    onClick={() => setImage(item.image)}
                    title="Нажми чтобы открыть"
                  />
                  <span className="history-tag">{item.model}</span>
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

export default App;
