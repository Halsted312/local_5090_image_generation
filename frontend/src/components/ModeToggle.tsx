type Mode = "generate" | "edit";

interface ModeToggleProps {
  mode: Mode;
  onChange: (mode: Mode) => void;
}

export default function ModeToggle({ mode, onChange }: ModeToggleProps) {
  return (
    <div className="mode-toggle">
      <button
        type="button"
        className={`mode-button ${mode === "generate" ? "active" : ""}`}
        onClick={() => onChange("generate")}
      >
        ✨ Generate
      </button>
      <button
        type="button"
        className={`mode-button ${mode === "edit" ? "active" : ""}`}
        onClick={() => onChange("edit")}
      >
        🛠️ Edit
      </button>
    </div>
  );
}
