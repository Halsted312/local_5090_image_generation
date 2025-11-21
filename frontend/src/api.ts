export interface TextGenerateRequest {
  prompt: string;
  num_inference_steps?: number;
  guidance_scale?: number;
  width?: number;
  height?: number;
  seed?: number | null;
}

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:6969";

async function handleResponse(res: Response): Promise<string> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed with status ${res.status}`);
  }
  const data = (await res.json()) as { image_base64: string };
  return data.image_base64;
}

export async function generateImage(req: TextGenerateRequest): Promise<string> {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  return handleResponse(res);
}
