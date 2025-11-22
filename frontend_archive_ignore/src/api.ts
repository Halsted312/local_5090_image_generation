export interface TextGenerateRequest {
  prompt: string;
  num_inference_steps?: number;
  guidance_scale?: number;
  width?: number;
  height?: number;
  seed?: number | null;
}

const API_BASE =
  (import.meta.env.VITE_API_URL ? String(import.meta.env.VITE_API_URL).replace(/\/$/, "") : "") || "";

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

export async function generatePrankImage(slug: string, req: TextGenerateRequest): Promise<string> {
  const res = await fetch(`${API_BASE}/api/p/${slug}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  return handleResponse(res);
}

// Prank builder APIs
export interface PrankCreateResponse {
  prank_id: string;
  slug: string;
  share_url: string;
}

export interface PrankDetail {
  prank_id: string;
  slug: string;
  title: string | null;
  triggers: { id: string; trigger_text: string; image_base64: string }[];
}

export async function fetchPrank(slug: string): Promise<PrankDetail> {
  const res = await fetch(`${API_BASE}/api/pranks/${slug}`);
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || "Failed to load prank");
  }
  return (await res.json()) as PrankDetail;
}

export async function createPrank(title?: string, slug?: string): Promise<PrankCreateResponse> {
  const res = await fetch(`${API_BASE}/api/pranks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title || null, slug: slug || null }),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || "Failed to create prank");
  }
  return (await res.json()) as PrankCreateResponse;
}

export async function addPrankTrigger(prankId: string, triggerText: string, file: File): Promise<void> {
  const form = new FormData();
  form.append("trigger_text", triggerText);
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/pranks/${prankId}/triggers`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || "Failed to add prank trigger");
  }
}
