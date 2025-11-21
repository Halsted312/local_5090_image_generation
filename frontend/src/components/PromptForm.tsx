interface PromptFormProps {
  prompt: string;
  steps: number;
  guidance: number;
  error?: string | null;
  isLoading: boolean;
  onPromptChange: (value: string) => void;
  onStepsChange: (value: number) => void;
  onGuidanceChange: (value: number) => void;
  onSubmit: () => void;
}

export default function PromptForm({
  prompt,
  steps,
  guidance,
  error,
  isLoading,
  onPromptChange,
  onStepsChange,
  onGuidanceChange,
  onSubmit,
}: PromptFormProps) {

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

      <div className="inputs-grid vertical" style={{ marginTop: "0.75rem" }}>
        <div className="field">
          <label htmlFor="steps">Detail</label>
          <small className="hint">1–10. More detail = slower.</small>
          <div className="slider-row">
            <input
              id="steps"
              type="range"
              min={1}
              max={10}
              value={steps}
              onChange={(e) => onStepsChange(Number(e.target.value))}
            />
            <div className="slider-badge">{steps}</div>
            <div className="slider-arrows">
              <button type="button" onClick={() => onStepsChange(Math.max(1, steps - 1))}>
                ◀
              </button>
              <button type="button" onClick={() => onStepsChange(Math.min(10, steps + 1))}>
                ▶
              </button>
            </div>
          </div>
        </div>
        <div className="field">
          <label htmlFor="guidance">Guidance</label>
          <small className="hint">1–3. Higher sticks closer to your words.</small>
          <div className="slider-row">
            <input
              id="guidance"
              type="range"
              min={1}
              max={3}
              value={guidance}
              onChange={(e) => onGuidanceChange(Number(e.target.value))}
            />
            <div className="slider-badge">{guidance}</div>
            <div className="slider-arrows">
              <button type="button" onClick={() => onGuidanceChange(Math.max(1, guidance - 1))}>
                ◀
              </button>
              <button type="button" onClick={() => onGuidanceChange(Math.min(3, guidance + 1))}>
                ▶
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="actions-row">
        <div className="action-left" />
        <div className="action-right">
          <button className="button" type="submit" disabled={isLoading}>
            {isLoading ? "Working..." : "Generate image"}
          </button>
          {error && <span className="error">{error}</span>}
        </div>
      </div>
    </form>
  );
}
