export interface GeneratedImage {
  id: string;
  src: string;
  prompt: string;
  mode?: "generate" | "prank";
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
          <span className={`image-badge ${img.mode === "prank" ? "badge-edit" : "badge-generate"}`}>
            {img.mode === "prank" ? "Prank" : "Generated"}
          </span>
          <div style={{ padding: "0.75rem", color: "#cbd5e1", fontSize: "0.9rem" }}>{img.prompt}</div>
        </div>
      ))}
    </div>
  );
}
