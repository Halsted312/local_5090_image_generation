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

export async function editImage(
  file: File,
  prompt: string,
  options?: { num_inference_steps?: number; guidance_scale?: number; seed?: number | null },
): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("prompt", prompt);
  if (options?.num_inference_steps !== undefined) {
    formData.append("num_inference_steps", String(options.num_inference_steps));
  }
  if (options?.guidance_scale !== undefined) {
    formData.append("guidance_scale", String(options.guidance_scale));
  }
  if (options?.seed !== undefined && options.seed !== null) {
    formData.append("seed", String(options.seed));
  }

  const res = await fetch(`${API_BASE}/api/edit`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(res);
}
