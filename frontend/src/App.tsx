import { useMemo, useState } from "react";
import ModeToggle from "./components/ModeToggle";
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
  const [prompt, setPrompt] = useState("");
  const [steps, setSteps] = useState(4);
  const [guidance, setGuidance] = useState(0);
  const [seed, setSeed] = useState<number | null>(null);
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

  const handleSubmit = async () => {
    setError(null);
    if (!prompt.trim()) {
      setError("Prompt is required.");
      return;
    }
    if (mode === "edit" && !file) {
      setError("Upload an image to edit.");
      return;
    }

    setIsLoading(true);
    try {
      let imageBase64: string;
      if (mode === "generate") {
        imageBase64 = await generateImage({
          prompt,
          num_inference_steps: steps,
          guidance_scale: guidance,
          width: 1024,
          height: 1024,
          seed: seed ?? undefined,
        });
      } else {
        imageBase64 = await editImage(file!, prompt, {
          num_inference_steps: steps,
          guidance_scale: guidance,
          seed,
        });
      }

      const newImage: GeneratedImage = {
        id: makeId(),
        src: imageBase64,
        mode,
        prompt,
      };
      setImages((prev) => [newImage, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="header">
        <div>
          <div className="pill">FLUX + FastAPI + Vite</div>
          <div className="title">Face Style Studio</div>
          <p style={{ color: "#9ca3af", marginTop: "0.3rem" }}>
            Upload a face, describe the body/look, and FLUX will craft the shot.
          </p>
        </div>
      </div>

      <div className="two-column">
        <div>
          <div className="section-title">Mode</div>
          <ModeToggle mode={mode} onChange={handleModeChange} />
          <div style={{ marginTop: "0.9rem" }}>
            <PromptForm
              mode={mode}
              prompt={prompt}
              steps={steps}
              guidance={guidance}
              seed={seed}
              file={file}
              error={error}
              isLoading={isLoading}
              onPromptChange={setPrompt}
              onStepsChange={setSteps}
              onGuidanceChange={setGuidance}
              onSeedChange={setSeed}
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
