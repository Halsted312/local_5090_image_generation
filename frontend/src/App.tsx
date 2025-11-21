import { useMemo, useState } from "react";
import PromptForm from "./components/PromptForm";
import ImageViewer, { GeneratedImage } from "./components/ImageViewer";
import { editImage, generateImage } from "./api";

type Mode = "generate" | "edit";

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export default function App() {
  const [mode, setMode] = useState<Mode>("generate");
  const [prompt, setPrompt] = useState("show me a cherry tree on a hill");
  const [steps, setSteps] = useState(6); // UI slider 1-10
  const [guidance, setGuidance] = useState(2); // UI slider 1-3
  const [file, setFile] = useState<File | null>(null);
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleModeChange = (nextMode: Mode) => {
    setMode(nextMode);
    setError(null);
    if (nextMode === "generate") {
      setFile(null);
    }
  };

  const statusText = useMemo(() => {
    if (isLoading) return "Working with FLUX...";
    if (error) return null;
    if (images.length > 0) return "Done. Add another variation or edit again.";
    return null;
  }, [error, images.length, isLoading]);

  const performRequest = async (opts?: { promptOverride?: string; modeOverride?: Mode }) => {
    const effectiveMode = opts?.modeOverride ?? mode;
    const effectivePrompt = opts?.promptOverride ?? prompt;
    setError(null);
    if (!effectivePrompt.trim()) {
      setError("Prompt is required.");
      return;
    }
    if (effectiveMode === "edit" && !file) {
      setError("Upload an image to edit.");
      return;
    }

    setIsLoading(true);
    try {
      let imageBase64: string;
      if (effectiveMode === "generate") {
        const mappedSteps = Math.round(4 + (steps - 1) * 0.5); // ~4–8.5
        const mappedGuidance = guidance === 1 ? 0 : guidance - 1; // 0–2
        imageBase64 = await generateImage({
          prompt: effectivePrompt,
          num_inference_steps: mappedSteps,
          guidance_scale: mappedGuidance,
          width: 1024,
          height: 1024,
        });
      } else {
        const mappedSteps = Math.min(40, Math.round(10 + (steps - 1) * 3)); // ~10–37
        const mappedGuidance = 1 + (guidance - 1) * 1.5; // 1.0–3.0
        imageBase64 = await editImage(file!, effectivePrompt, {
          num_inference_steps: mappedSteps,
          guidance_scale: mappedGuidance,
        });
      }

      const newImage: GeneratedImage = {
        id: makeId(),
        src: imageBase64,
        mode: effectiveMode,
        prompt: effectivePrompt,
      };
      setImages((prev) => [newImage, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    await performRequest();
  };

  return (
    <div className="page">
      <div className="header" style={{ flexDirection: "column", alignItems: "center" }}>
        <div className="title">Imgen 4 U</div>
        <p style={{ color: "#9ca3af", marginTop: "0.2rem", textAlign: "center" }}>
          Describe a look or upload a photo to transform it. GPU-powered edits and generations.
        </p>
      </div>

      <div className="layout">
        <div className="panel">
          <div className="section-title">Choose how to work</div>
          <div className="mode-toggle">
            <button
              type="button"
              className={`mode-button ${mode === "generate" ? "active" : ""}`}
              onClick={() => handleModeChange("generate")}
            >
              ✨ Generate from text
            </button>
            <button
              type="button"
              className={`mode-button ${mode === "edit" ? "active" : ""}`}
              onClick={() => handleModeChange("edit")}
            >
              📤 Upload photo
            </button>
          </div>
          <div style={{ marginTop: "0.9rem" }}>
            <PromptForm
              mode={mode}
              prompt={prompt}
              steps={steps}
              guidance={guidance}
              file={file}
              error={error}
              isLoading={isLoading}
              onPromptChange={setPrompt}
              onStepsChange={setSteps}
              onGuidanceChange={setGuidance}
              onFileChange={setFile}
              onSubmit={handleSubmit}
            />
            {statusText && <div className="status" style={{ marginTop: "0.5rem" }}>{statusText}</div>}
          </div>
        </div>

        <div className="panel">
          <div className="section-title">Output</div>
          <ImageViewer images={images} />
        </div>
      </div>
    </div>
  );
}
