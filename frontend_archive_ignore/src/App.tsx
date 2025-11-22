import { useMemo, useState } from "react";
import PromptForm from "./components/PromptForm";
import ImageViewer, { GeneratedImage } from "./components/ImageViewer";
import { generateImage } from "./api";

function makeId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export default function App() {
  const [prompt, setPrompt] = useState("show me a cherry tree on a hill");
  const [steps, setSteps] = useState(6); // UI slider 1-10
  const [guidance, setGuidance] = useState(2); // UI slider 1-3
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const statusText = useMemo(() => {
    if (isLoading) return "Working with FLUX...";
    if (error) return null;
    if (images.length > 0) return "Done. Add another variation.";
    return null;
  }, [error, images.length, isLoading]);

  const performRequest = async (opts?: { promptOverride?: string }) => {
    const effectivePrompt = opts?.promptOverride ?? prompt;
    setError(null);
    if (!effectivePrompt.trim()) {
      setError("Prompt is required.");
      return;
    }

    setIsLoading(true);
    try {
      const mappedSteps = Math.round(4 + (steps - 1) * 0.5); // ~4–8.5
      const mappedGuidance = guidance === 1 ? 0 : guidance - 1; // 0–2
      const imageBase64 = await generateImage({
        prompt: effectivePrompt,
        num_inference_steps: mappedSteps,
        guidance_scale: mappedGuidance,
        width: 640,
        height: 640,
      });

      const newImage: GeneratedImage = {
        id: makeId(),
        src: imageBase64,
        prompt: effectivePrompt,
        mode: "generate",
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
        <div className="title">Prompt Pics</div>
        <p style={{ color: "#9ca3af", marginTop: "0.2rem", textAlign: "center" }}>
          Describe a look and FLUX will paint it from scratch. GPU-powered generations only.
        </p>
      </div>

      <div className="layout">
        <div className="panel">
          <div className="section-title">Text → Image</div>
          <PromptForm
            prompt={prompt}
            steps={steps}
            guidance={guidance}
            error={error}
            isLoading={isLoading}
            onPromptChange={setPrompt}
            onStepsChange={setSteps}
            onGuidanceChange={setGuidance}
            onSubmit={handleSubmit}
          />
          {statusText && <div className="status" style={{ marginTop: "0.5rem" }}>{statusText}</div>}
        </div>

        <div className="panel">
          <div className="section-title">Output</div>
          <ImageViewer images={images} />
        </div>
      </div>
    </div>
  );
}
