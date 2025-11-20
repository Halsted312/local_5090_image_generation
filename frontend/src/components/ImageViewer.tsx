type Mode = "generate" | "edit";

export interface GeneratedImage {
  id: string;
  src: string;
  mode: Mode;
  prompt: string;
}

interface ImageViewerProps {
  images: GeneratedImage[];
}

export default function ImageViewer({ images }: ImageViewerProps) {
  if (images.length === 0) {
    return <p style={{ color: "#cbd5e1" }}>No images yet — craft a prompt and go.</p>;
  }

  return (
    <div className="image-grid">
      {images.map((img) => (
        <div className="image-card" key={img.id}>
          <img src={`data:image/png;base64,${img.src}`} alt={img.prompt} loading="lazy" />
          <span className={`image-badge ${img.mode === "generate" ? "badge-generate" : "badge-edit"}`}>
            {img.mode === "generate" ? "Generated" : "Edited"}
          </span>
        </div>
      ))}
    </div>
  );
}
