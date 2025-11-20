import { useEffect, useMemo } from "react";

type Mode = "generate" | "edit";

interface PromptFormProps {
  mode: Mode;
  prompt: string;
  steps: number;
  guidance: number;
  seed: number | null;
  file: File | null;
  error?: string | null;
  isLoading: boolean;
  onPromptChange: (value: string) => void;
  onStepsChange: (value: number) => void;
  onGuidanceChange: (value: number) => void;
  onSeedChange: (value: number | null) => void;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void;
}

export default function PromptForm({
  mode,
  prompt,
  steps,
  guidance,
  seed,
  file,
  error,
  isLoading,
  onPromptChange,
  onStepsChange,
  onGuidanceChange,
  onSeedChange,
  onFileChange,
  onSubmit,
}: PromptFormProps) {
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <form
      className="panel"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <div className="field">
        <label htmlFor="prompt">Prompt</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          placeholder="Describe the scene, style, or changes you want..."
        />
      </div>

      <div className="inputs-grid" style={{ marginTop: "0.75rem" }}>
        <div className="field">
          <label htmlFor="steps">Steps</label>
          <input
            id="steps"
            type="number"
            min={1}
            max={50}
            value={steps}
            onChange={(e) => onStepsChange(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="guidance">Guidance</label>
          <input
            id="guidance"
            type="number"
            step="0.1"
            min={0}
            max={50}
            value={guidance}
            onChange={(e) => onGuidanceChange(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="seed">Seed (optional)</label>
          <input
            id="seed"
            type="number"
            value={seed ?? ""}
            onChange={(e) => onSeedChange(e.target.value === "" ? null : Number(e.target.value))}
            placeholder="Random if blank"
          />
        </div>
      </div>

      {mode === "edit" && (
        <div className="field" style={{ marginTop: "0.75rem" }}>
          <label htmlFor="file">Upload image to edit</label>
          <input
            id="file"
            type="file"
            accept="image/*"
            onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
          />
          {file && (
            <div className="file-info">
              <span>Selected: {file.name}</span>
            </div>
          )}
          {previewUrl && (
            <div className="file-preview">
              <img src={previewUrl} alt="Preview" />
            </div>
          )}
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginTop: "1rem" }}>
        <button className="button" type="submit" disabled={isLoading}>
          {isLoading ? "Working..." : mode === "generate" ? "Generate image" : "Edit image"}
        </button>
        {error && <span className="error">{error}</span>}
      </div>
    </form>
  );
}
