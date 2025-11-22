Backend Architecture and Implementation Plan for AI Image Generator and Prank Service
System Overview
This backend will be a FastAPI server running on a powerful local GPU machine (e.g. with an NVIDIA 5090, 128GB RAM). It exposes a RESTful API that the Replit-hosted frontend can communicate with (likely via an ngrok URL). The server has no user accounts or authentication; instead it uses random slug identifiers for managing user-created “prank sets” and their public share links. The core features include:
    • AI Image Generation from text prompts using multiple models (Stable Diffusion variants and others).
    • Prank Prompt Sets where a user can upload custom image outputs for specific trigger prompts (accessible via unique slugs).
    • Smart Model Routing using a local LLM (Llama 3 8B instruct) to choose the best generation model for a given prompt (and to detect prank trigger similarity).
    • Logging and Versioning of all generations and edits, stored in a database (PostgreSQL if available, or SQLite fallback) and on disk.
The backend will handle heavy tasks (model inferences, image processing) on the GPU machine, while the Replit front-end simply calls the backend API endpoints to generate images, manage prank sets, and fetch logs.
Tech Stack and Components
    • Language & Framework: Python 3.11+ with FastAPI for building the REST API.
    • Database: PostgreSQL (preferred for production) with SQLAlchemy, falling back to SQLite for simplicity if needed (file-based for local testing).
    • ML Models: Multiple image generation models and one routing LLM:
    • Image Generation Models:
        ◦ FLUX.1-dev – general-purpose image generator (e.g. a Stable Diffusion derivative for all-round use).
        ◦ RealVisXL V4.0 – photorealistic model (great for faces/portraits).
        ◦ SD3-Medium – Stable Diffusion 3 (medium-sized) for complex compositions, UI elements, or multi-part scenes.
        ◦ SDXL Base (logo_sdxl) – Stable Diffusion XL base model, used as a fallback especially for text/logos clarity.
    • Routing LLM: Llama 3 8B Instruct – a local large language model used to analyze prompts and decide which image model to use (and to detect prank trigger matches). This model will produce a JSON decision including chosen model, confidence scores, reasoning, tags, and any prank-match info.
    • Image Processing: PIL (Pillow) for image format conversion and thumbnail generation. Images will typically be generated as PNG for full quality and saved to disk, with JPEG/WebP thumbnails for quick previews.
    • Server Infrastructure: The FastAPI app will run with Uvicorn (or Hypercorn) and likely leverage the GPU via libraries like Hugging Face diffusers for Stable Diffusion and llama.cpp or Hugging Face Transformers for the LLM. We will load models at startup and keep them in memory for performance[1][2]. Long-running tasks (image generation) will be handled in worker threads or background tasks to avoid blocking the event loop[3].
    • Ngrok Tunneling: The local server will be exposed to the internet (and the Replit frontend) via an ngrok tunnel. The frontend will hit endpoints like https://<ngrok-id>.ngrok.io/api/generate transparently.
Database Design
Using an SQL database ensures persistence of prompts, images, and logs. Below is the schema with three main tables and their fields:
    • prank_sets – each “prank set” (a collection of trigger prompts and user-uploaded images):
    • id (PK, serial): Internal ID for the prank set.
    • creator_slug (TEXT, unique): The private slug used by the creator to edit this set (e.g. "9xg2kt"). Generated randomly; not easily guessable.
    • share_slug (TEXT, unique): The public slug used to share the prank link (e.g. "82f6qk"). Also random 6 characters.
    • title (TEXT, nullable): Optional title or label for the prank set (for UI purposes).
    • created_at (TIMESTAMP): Creation timestamp.
    • prank_prompts – the prompt-image pairs within a prank set:
    • id (PK, serial): ID of the prank prompt entry.
    • prank_set_id (FK → prank_sets.id): Reference to which prank set this belongs.
    • trigger_prompt (TEXT): The exact text of the trigger prompt that should activate this prank.
    • image_path (TEXT): Filesystem path to the full image uploaded for this trigger.
    • thumbnail_path (TEXT): Path to a smaller thumbnail image.
    • created_at (TIMESTAMP): Timestamp when this prompt+image was added.
    • generation_logs – log of all image generations and edits (both normal and via prank links):
    • id (PK, serial): ID of the generation event.
    • prompt (TEXT): The text prompt that was used for generation.
    • model_id (TEXT): The ID of the model used to generate the image. (If a prank was used instead of model generation, this could be a special value like "prank" or left NULL).
    • router_json (JSONB/TEXT): The full JSON output from the router LLM describing the decision (model scores, tags, reasons, etc)[1].
    • image_path (TEXT): Filesystem path to the generated (or served) image.
    • thumbnail_path (TEXT): Path to the thumbnail image.
    • prank_set_id (INT, nullable): If this generation was from a prank link and matched a prank, reference which prank set’s image was used.
    • creator_slug (TEXT, nullable): If the generation was initiated in the context of an editing session (the prank set creator’s interface), store the creator’s slug (for example, if the creator tested generating something while customizing – though primarily, creators upload images rather than generate in that flow).
    • share_slug (TEXT, nullable): If the generation was initiated via a public share link, store that slug (for analytics or for later tie-back).
    • version_number (INT): Version number of the image if it was edited. Initial generation = 1; any user edits increment this.
    • edited_from_generation_id (INT, nullable): If this image is an edit of a previous generation, this is a reference to that original generation log entry.
    • created_at (TIMESTAMP): Generation timestamp.
Note on JSON storage: In PostgreSQL, router_json can be a JSONB column for efficient querying. In SQLite (which lacks native JSON type), we can store it as TEXT (stringified JSON).
ORMs/Pydantic: We can define SQLAlchemy models for these tables and Pydantic schemas for input/output. For example, using SQLAlchemy:
# Example SQLAlchemy model definitions
class PrankSet(Base):
    __tablename__ = 'prank_sets'
    id = Column(Integer, primary_key=True)
    creator_slug = Column(String(6), unique=True, nullable=False)
    share_slug = Column(String(6), unique=True, nullable=False)
    title = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class PrankPrompt(Base):
    __tablename__ = 'prank_prompts'
    id = Column(Integer, primary_key=True)
    prank_set_id = Column(Integer, ForeignKey('prank_sets.id', ondelete='CASCADE'))
    trigger_prompt = Column(Text, nullable=False)
    image_path = Column(Text, nullable=False)
    thumbnail_path = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class GenerationLog(Base):
    __tablename__ = 'generation_logs'
    id = Column(Integer, primary_key=True)
    prompt = Column(Text, nullable=False)
    model_id = Column(String, nullable=True)
    router_json = Column(Text, nullable=True)
    image_path = Column(Text, nullable=False)
    thumbnail_path = Column(Text, nullable=False)
    prank_set_id = Column(Integer, ForeignKey('prank_sets.id'), nullable=True)
    creator_slug = Column(String(6), nullable=True)
    share_slug = Column(String(6), nullable=True)
    version_number = Column(Integer, default=1)
    edited_from_generation_id = Column(Integer, ForeignKey('generation_logs.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
We would also set up indices on the slug fields (since we will frequently query by slug) and possibly on created_at for sorting logs.
File Storage and Directory Structure
All images and thumbnails will be saved on disk in an organized folder structure. Since this is a single-instance server, a simple approach is fine:
    • Base image directory: e.g. ./data/images/ (ensuring this folder exists on disk).
    • Within this, we can have subfolders for different types of images:
        ◦ data/images/generations/ – for AI-generated images (by the models).
        ◦ data/images/pranks/ – for images uploaded as part of prank sets.
        ◦ Optionally, separate subfolders for thumbnails (or we can store thumbs alongside originals with different file extension).
For simplicity, we might store thumbnails in the same directory with a suffix in the filename. For example, if a generated image is saved as data/images/generations/gen_42.png, its thumbnail could be data/images/generations/gen_42_thumb.jpg. Similarly, a prank image prank_7.png might have prank_7_thumb.jpg. We ensure file names are unique to avoid collisions: - Generated images could be named by their generation_log ID (gen_{id}.png). - Prank images could be named by their prank_prompts ID (prank_{id}.png). - Alternatively, use UUIDs or the slug plus an index for filenames. (Using the database IDs is straightforward since they are unique primary keys).
Static File Serving: We will configure FastAPI to serve the data/images/ directory via StaticFiles. This allows the frontend to access image URLs directly if needed. For example, mounting static files:
from fastapi.staticfiles import StaticFiles
app.mount("/images", StaticFiles(directory="data/images"), name="images")
This means any image_path or thumbnail_path stored (which will be relative to data/images) can be served by constructing a URL like http://<server>/images/generations/gen_42.png. The Replit frontend can then display images by their URLs. We’ll include these URLs in API responses where appropriate (for instance, when returning generation logs or prank set details).
Slug Generation for Prank Sets
When a user clicks "Customize" on the frontend, the backend needs to create a new prank set with two unique 6-character slugs: one for the creator and one for sharing. We will generate these slugs using a secure random generator to avoid predictable sequences. A common approach is to use Python’s secrets module to get a random string of alphanumeric characters[4]:
import secrets, string

def generate_slug(length=6) -> str:
    alphabet = string.ascii_lowercase + string.digits  # 26 letters + 10 digits = 36 chars
    return ''.join(secrets.choice(alphabet) for _ in range(length))
We will ensure uniqueness by checking the database (if a collision is found – extremely rare with 36^6 ≈ 2 billion possibilities – we generate a new one). Each slug will be stored in the prank_sets table. The creator_slug is meant to be kept private (only shown to the user who created the prank set, perhaps in their Replit session). The share_slug is meant to be given out publicly. Both are unguessable in practice due to their randomness and length.
Model Loading and Routing Engine
Image Generation Models
We will load each of the four image generation models at application startup, so that subsequent requests can use them without reloading. Given the large size of these models, we need to consider memory usage. If VRAM allows, we can keep them loaded on the GPU; otherwise, we might keep some on CPU and move to GPU as needed. For the initial design, we assume the machine can handle them (possibly the 5090 has ample VRAM) and load all to GPU for quick switching.
Pseudocode for loading models using Hugging Face diffusers (as an example) at startup:
from diffusers import StableDiffusionPipeline
# ... within startup event or similar:
model_pipes = {}
model_pipes["flux_dev"] = StableDiffusionPipeline.from_pretrained("path_or_id_of_FLUX.1-dev", torch_dtype=torch.float16).to("cuda")
model_pipes["realvis_xl"] = StableDiffusionPipeline.from_pretrained("path_or_id_of_RealVisXL_V4", torch_dtype=torch.float16).to("cuda")
model_pipes["sd3_medium"] = StableDiffusionPipeline.from_pretrained("path_or_id_of_SD3-Medium", torch_dtype=torch.float16).to("cuda")
model_pipes["logo_sdxl"] = StableDiffusionPipeline.from_pretrained("path_or_id_of_SDXL-Base", torch_dtype=torch.float16).to("cuda")
(The actual model identifiers or local paths need to be provided. The above assumes we have the model weights available. If these are custom models not on HuggingFace, we’d load from local checkpoints.)
We will also set any schedulers or specific settings as needed (e.g. diffusers usually lets you swap schedulers like Euler vs. DDIM if needed). The initialization happens once. For example, loading a Stable Diffusion pipeline and moving it to GPU[1]:
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
We also consider performance: running multiple model inferences concurrently on one GPU might slow things down or run out of memory. To manage this, we will queue or serialize generation requests. A simple way is to use a single ThreadPoolExecutor with max_workers=1 (one generation at a time)[5], or use an async task queue. Given the likely usage, a single thread off the main event loop can process image generations sequentially, which avoids overloading the GPU with concurrent runs. FastAPI’s dependency system or background tasks can help here, but a global executor is straightforward.
Example using ThreadPoolExecutor for a generation call[3]:
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)

# Later, in the generate endpoint:
result = executor.submit(model_pipes[selected_model].__call__, prompt, **pipeline_params).result()
image = result.images[0]  # PIL Image
After generation, we will have a PIL Image object. We then save it to disk (PNG format for full quality), generate a thumbnail, and possibly also prepare it for response. We should also free up memory if needed by calling torch.cuda.empty_cache() after heavy operations (as shown in similar implementations[3]).
Router LLM (Model Selection and Prank Matching)
We will integrate the Llama 3 8B Instruct model as our routing engine. This model will run locally (likely via CPU or quantized on GPU). We can use the llama-cpp-python library or HuggingFace Transformers with an appropriate model. The model is loaded at startup and kept in memory (since 8B is manageable in 128GB RAM, possibly in 8-bit mode it might even fit on GPU). For example, using llama-cpp-python we could load a GGUF or GGML model file:
from llama_cpp import Llama
router_model = Llama(model_path="/models/llama-3b-8binstruct.gguf", n_threads=16)
# Alternatively, using huggingface:
# router_model = transformers.AutoModelForCausalLM.from_pretrained("Llama-3-8b-instruct", ... )
We will ensure the model is loaded once on startup[2]. Using FastAPI’s startup events or lifespan context is appropriate: we can store the loaded model in app.state.router_model for use in endpoints[2].
Routing Logic and Prompting
The router’s job is to examine the input prompt (and optionally a list of prank trigger prompts) and output a JSON with the following schema:
{
  "model_id": "<best_model_choice>",
  "scores": { "<model_id>": <score>, ... }, 
  "tags": [ "<tag1>", "<tag2>", ... ],
  "reason": "<brief explanation>",
  "matched_prank_id": <ID if a prank trigger matched, otherwise null or not present>
}
Model Selection Criteria: We will instruct the LLM with a system prompt or few-shot examples to apply the given rules: - If the prompt explicitly mentions or implies a portrait or human face, strongly favor realvis_xl (add +1.5 weight to its score). - If the prompt involves logos, icons, or text content within the image, favor logo_sdxl (add +1.2). - If the prompt is complex, multi-part (for example, long prompts with multiple sentences, or mentions of UI elements, posters, etc.), favor sd3_medium (+1.1). - Otherwise, flux_dev serves as the general fallback model (it might get a base weight, or simply end up highest if others have no boosts).
The LLM will consider these factors and produce a confidence score for each model, then pick the highest. It will also output tags which are keywords it noticed (e.g. it might output ["portrait"] if it saw words like "person" or "face" in the prompt, etc., which justify why it chose a model). The reason field is a human-readable short explanation for logging/debugging (e.g. "Prompt describes a face, using RealVisXL for photorealism").
Prank Matching: If a list of prank trigger prompts is provided to the router, it should also assess whether the user’s prompt is semantically similar to any of them. The similarity check can be done by the LLM’s understanding (we provide the list in the prompt context). If the prompt is essentially the same request as one of the triggers (or very close), the router should set matched_prank_id to that trigger’s ID. This would indicate that instead of using an AI model, we should serve the pre-uploaded prank image.
To accomplish this, we craft the LLM prompt as follows (pseudocode for the prompt to Llama):
System: You are a routing assistant for an AI image generator. 
Given a user prompt and some available image models, you will choose the best model or detect a prank trigger.
Follow these rules for model selection:
- If prompt mentions portrait or human faces, RealVisXL gets +1.5 weight.
- If prompt mentions logo, icon, or text in image, SDXL (logo_sdxl) gets +1.2 weight.
- If prompt is very complex or multi-scene, SD3-Medium gets +1.1 weight.
- Otherwise, default to FLUX.1-dev.
Also, you may be provided with prank trigger prompts. If the user's prompt is very similar in meaning to any prank trigger, output "matched_prank_id" with that trigger's ID.

User Prompt: "<the actual user prompt text>"

Available Models:
- flux_dev: (general-purpose image model)
- realvis_xl: (photorealistic model for faces and people)
- sd3_medium: (complex composition model)
- logo_sdxl: (specialized for logos/text)

Prank Triggers (if any):
1. "<Trigger Prompt 1>" (id=<ID1>)
2. "<Trigger Prompt 2>" (id=<ID2>)
... (list all triggers with their IDs)

Now, analyze the user prompt. Determine an appropriate score for each model (base 0-1 scale) after applying any weights. Decide the best model. Also check triggers for similarity.
Respond **ONLY** with a JSON object, with keys:
"model_id", "scores", "tags", "reason", and if a prank matches, "matched_prank_id".
The LLM should output a JSON string (we will have to parse it with json.loads). We might use a library like pydantic or manually verify the keys. If the JSON is malformed, we handle exceptions (perhaps retry or default to flux_dev to be safe).
Example: For prompt "A realistic photo of an old man smiling, detailed face, 8K" the router LLM might produce:
{
  "model_id": "realvis_xl",
  "scores": {
    "flux_dev": 0.2,
    "realvis_xl": 0.95,
    "sd3_medium": 0.1,
    "logo_sdxl": 0.0
  },
  "tags": ["portrait", "face", "photorealistic"],
  "reason": "The prompt describes a human face in realistic detail, so RealVisXL is most suitable.",
  "matched_prank_id": null
}
If the prompt was very close to a prank trigger, say the trigger "Why did the chicken cross the road?" and the user prompt is "Why would a chicken cross the road?", the LLM might output "matched_prank_id": 42 (with 42 being the ID of that prank trigger) along with maybe still suggesting a model in case (but we will prioritize serving the prank image if a match is found).
Routing Model Inference: To implement the above, we use the loaded LLM. Using llama_cpp as an example, after constructing the prompt text we can do:
response = app.state.router_model(prompt_text, stop=["}"], max_tokens=256)  # we stop at `}` assuming well-formed JSON
router_out = response["choices"][0]["text"]
decision = json.loads(router_out)
If using HuggingFace transformers, we’d use generate() on the model with proper stopping conditions or parse the text. We must be careful to instruct the model to output only JSON. We might use few-shot examples or system messages (if using an instruct model) to reinforce that.
API Endpoints and Implementation
Now, we detail each API endpoint, including their purpose, input/output schema, and any important logic. FastAPI allows us to define these with decorators. We will include example request/response structures for clarity.
1. POST /api/generate – Generate an image from a prompt
Description: Main endpoint to handle prompt-to-image generation. The request can either let the router decide the best model (“engine”: "auto") or force a particular model. A new image is generated and saved, and a log entry created.
    • Request Body: JSON with fields:
    • prompt (string, required): The text prompt describing the desired image.
    • engine (string, optional): Which model to use. Can be one of "auto", "flux_dev", "realvis_xl", "sd3_medium", "logo_sdxl". Default "auto" means use the router LLM to pick.
    • Response: On success, returns JSON with details of the generation:
    • generation_id: The log ID of this generation.
    • model_id: The model that was used (or "prank" if somehow a prank was triggered here, though this endpoint won't normally involve pranks).
    • image_url: URL path to the generated image (PNG) – e.g. "/images/generations/gen_45.png".
    • thumbnail_url: URL to the thumbnail (e.g. "/images/generations/gen_45_thumb.jpg").
    • router_metadata: The router JSON (as an object) describing the decision (only present if engine="auto").
    • created_at: timestamp.
Logic: 1. Parse the input JSON. If engine is not provided or is "auto", use the router LLM:
if engine == "auto":
    decision = route_model(prompt)
    selected_model = decision["model_id"]
else:
    selected_model = engine
    decision = None
(Here route_model(prompt) encapsulates the prompt construction and LLM call described earlier, without prank triggers.) 2. Load the appropriate model pipeline from our model_pipes dict. (They’re already in memory.) 3. Trigger image generation. This could be a heavy operation, so do it in a thread worker to not block the main loop[3]. For example:
image = executor.submit(model_pipes[selected_model], prompt).result().images[0]
We might allow additional parameters (like guidance scale, steps, etc.) but those can also be defaulted for simplicity or taken from the request if needed. 4. Once the PIL image is obtained, save it to disk: - Determine the next generation ID (if using DB auto-increment, we may need to insert the DB row after saving to get the ID, or use a temporary UUID name). We can also insert a log entry before generation to reserve an ID. For simplicity, we’ll insert after generation: - Create a filename, e.g. use a UUID or use a placeholder and get the ID later. For now, we can save to a temp file.
import uuid
filename = f"{uuid.uuid4().hex}.png"
save_path = os.path.join("data/images/generations", filename)
image.save(save_path, format="PNG")
- Create thumbnail (say 256px max dimension):
thumb = image.copy()
thumb.thumbnail((256, 256))
thumb_filename = f"{filename.rsplit('.',1)[0]}_thumb.jpg"
thumb_path = os.path.join("data/images/generations", thumb_filename)
thumb.save(thumb_path, format="JPEG", quality=80)
Alternatively, use PIL’s Image.thumbnail method as above or any resizing. We ensure mode is RGB for JPEG. (We could offload thumbnail generation to a background task to shave off a bit of latency, returning the main image immediately, but since generation is the bottleneck, thumbnailing is quick. FastAPI’s BackgroundTasks could be used if needed[6].) 5. Insert a row into generation_logs with all details: - prompt, model_id, router_json (as text or JSON), image_path (relative path like "generations/<file>.png"), thumbnail_path ("generations/<file>_thumb.jpg"), prank_set_id=NULL, creator_slug=NULL, share_slug=NULL, version_number=1, edited_from_generation_id=NULL, created_at=now(). - Get the generated id (generation_id). 6. Return the response JSON as described. (We might also directly return the image in the response for immediate display, but since we want to log and possibly allow re-fetching, returning a URL and metadata is cleaner. If immediate image display is needed, the frontend can use the image_url or we could provide an option like ?stream=true to stream the image.)
Example: Request:
POST /api/generate
{
  "prompt": "A fantasy castle on a hill during sunset, digital art",
  "engine": "auto"
}
Possible Response:
{
  "generation_id": 57,
  "model_id": "flux_dev",
  "image_url": "/images/generations/gen_57.png",
  "thumbnail_url": "/images/generations/gen_57_thumb.jpg",
  "router_metadata": {
      "model_id": "flux_dev",
      "scores": { "flux_dev": 0.8, "realvis_xl": 0.1, "sd3_medium": 0.4, "logo_sdxl": 0.0 },
      "tags": ["landscape"],
      "reason": "No specific indicators for other models; using general model."
  },
  "created_at": "2025-11-21T18:25:53Z"
}
(If engine was set to a specific model, router_metadata would be omitted or null.)
2. POST /api/prank-sets – Create a new prank set
Description: Generates a new prank set entry with unique slugs. This is called when a user initiates the prank customization process. No authentication; the returned slugs secure the access.
    • Request Body: JSON (optional fields):
    • title (string, optional): An optional title for the prank set (e.g. "John’s Birthday Prank").
    • Response: JSON with:
    • prank_set_id: The internal ID of the new prank set (could be omitted if we prefer to use slugs externally, but useful for reference).
    • creator_slug: The private slug (to be used in URLs for editing this set).
    • share_slug: The public slug (to be used for sharing the prank link).
    • created_at: timestamp.
Logic: 1. Generate a new creator_slug and share_slug (6 chars each) using the random function described earlier, ensuring they don’t conflict with existing slugs in prank_sets[4]. 2. Insert a new prank_sets row with these slugs, the provided title (or NULL), and current timestamp. 3. Return the slugs and id.
Example: Request:
POST /api/prank-sets
{ "title": "My Prank Set" }
Response:
{
  "prank_set_id": 12,
  "creator_slug": "9xg2kt",
  "share_slug": "82f6qk",
  "created_at": "2025-11-21T18:26:10Z"
}
The frontend will likely redirect the user to a customization page like /custom/9xg2kt and use the creator_slug to fetch/update the prank set. The share_slug can be shown so the user knows the link to share (e.g. promptpics.ai/p/82f6qk).
3. POST /api/prank-sets/{creator_slug}/prompts – Add a trigger prompt & image to a prank set
Description: Adds a new trigger prompt and its corresponding image to an existing prank set. This endpoint handles file upload.
    • URL Path: {creator_slug} is the private slug identifying the prank set (we use this instead of numeric ID for security by obscurity).
    • Request: This is a multipart form request (since it includes an image file):
    • trigger_prompt – form field (text) for the trigger prompt.
    • image – form field (file upload) for the image to associate with this prompt.
    • Response: JSON with the created prank prompt entry:
    • prompt_id: The ID of the newly created prank_prompts entry.
    • trigger_prompt: (echo back the text).
    • image_url: path to the uploaded image (for preview, e.g. /images/pranks/prank_5.png).
    • thumbnail_url: path to the thumbnail of the uploaded image.
    • created_at: timestamp.
Logic: 1. Verify the prank set exists by looking up creator_slug in prank_sets. If not found, return 404. 2. Read the trigger_prompt text from form data. 3. Read the uploaded image file. We can use UploadFile from FastAPI which gives us a file-like object[7]. For example:
file: UploadFile
image = Image.open(file.file)
We might want to ensure it’s in a standard format (convert to RGB to drop alpha, etc.)[8]:
if image.mode in ("RGBA", "P"):
    image = image.convert("RGB")
4. Save the image to the data/images/pranks/ directory. We can name it prank_<prank_prompt_id>.png, but since we don’t have the ID yet, consider using a temp name or using the next sequence number: - We can get the prank_set_id from the looked-up prank set. - Save the image first to a temporary file:
filename = f"{uuid.uuid4().hex}.png"
save_path = os.path.join("data/images/pranks", filename)
image.save(save_path, format="PNG")
5. Create a thumbnail similarly (e.g. max 256x256):
thumb = image.copy()
thumb.thumbnail((256, 256))
thumb_filename = f"{filename.rsplit('.',1)[0]}_thumb.jpg"
thumb_path = os.path.join("data/images/pranks", thumb_filename)
thumb.save(thumb_path, format="JPEG")
6. Insert a new row into prank_prompts with prank_set_id, trigger_prompt, and the image_path (e.g. "pranks/<filename>") and thumbnail_path ("pranks/<thumb_filename>"), plus timestamp. 7. Now we have the id of the prank prompt (primary key). If we want, we could rename the files to use this id (e.g. prank_<id>.png). This is optional; if we do, we should also update the DB entry with the new name. Alternatively, we can simply keep the UUID name or whatever was used (and the DB references it). 8. Return the response with prompt details. The image_url would be assembled as "/images/pranks/<filename>" and thumbnail similar.
Example: Request: (multipart form)
POST /api/prank-sets/9xg2kt/prompts
Form fields: trigger_prompt="Why did the chicken cross the road?"
File: image=@chicken_meme.png
Response:
{
  "prompt_id": 34,
  "trigger_prompt": "Why did the chicken cross the road?",
  "image_url": "/images/pranks/prank_34.png",
  "thumbnail_url": "/images/pranks/prank_34_thumb.jpg",
  "created_at": "2025-11-21T18:30:45Z"
}
(Where "prank_34.png" is the stored file, renamed with ID 34 for clarity.)
4. GET /api/p/{share_slug}/generate – Generate or retrieve image via a prank share link
Description: This is the endpoint that powers the public prank link. It takes a user-provided prompt and either returns a prank image (if the prompt matches one of the triggers in that prank set) or generates a fresh AI image (using the router and models as usual) if there’s no trigger match. This allows a single share link to either show a pre-defined image (for specific trigger words) or behave like a normal AI generator for other prompts, making the prank less suspicious.
    • URL Path: {share_slug} is the public slug identifying which prank set to use.
    • Query Parameter: prompt (string, required) – the prompt input by the user on the share page.
    • Response: This will return an image directly (most likely). We have two scenarios:
    • If a prank trigger matches: return the prank image (the one the prank creator uploaded) as the response.
    • If no match: generate a new image using the same process as /api/generate and return that.
Because the frontend might be expecting an image (to directly display it), we can return a binary image response (e.g. image/png content). However, we also want to log this event. So we will perform the logic internally and then stream or send the image.
Logic: 1. Look up the prank set by share_slug. If not found, return 404. 2. Retrieve all trigger prompts for this prank set (you could query prank_prompts by prank_set_id). If there are none (empty set), then this share link isn’t configured – we can just treat it as no match (generate normally). 3. Use the router LLM to check for a match. We can actually use the same router function, passing in the list of trigger prompts:
decision = route_model(user_prompt, prank_triggers=list_of_trigger_texts)
The LLM will include matched_prank_id if it finds a close match. Alternatively, a simpler approach is to manually check similarity (e.g. exact match or embedding similarity). But since we want the LLM’s semantic power, we go with that. 4. If decision["matched_prank_id"] exists (not null): - Find the corresponding prank prompt entry (by ID). - Get the image_path from it (the stored prank image). - Open the image file from disk. - Return the image file content directly. In FastAPI, we can use FileResponse or StreamingResponse. For example:
from fastapi.responses import FileResponse
return FileResponse(path="data/images/pranks/prank_34.png", media_type="image/png")
- Also log this event in generation_logs: - prompt = the user prompt, - model_id = maybe null or "prank", - router_json = the decision JSON (which includes the prank match info and possibly model scores), - image_path = that prank image path, thumbnail_path = its thumb, - prank_set_id = that prank set’s id, - share_slug = the share slug, - version_number = 1, edited_from_generation_id = NULL. - No new image was generated, so no new files need saving (just serving existing one). - Return the image response. 5. If no match (matched_prank_id is not present or null): - We proceed to generate an image using the selected model from decision (or if we chose not to run router for this? But likely we still use router to pick best model for the prompt). - Use decision["model_id"] as selected model (if we ran the router; if we skip router for prank, we could directly route anyway – but better to maintain consistent behavior and also use router for model selection). - Generate the image with that model (same as /api/generate steps). - Save image and thumbnail to data/images/generations/ (or we might choose a separate folder for share generations, but not necessary). - Insert into generation_logs with share_slug (and prank_set_id could be recorded as well or left null because no prank triggered; however, we know which prank set was used even if no specific prank matched, so we might still log the prank_set_id for context). - Return the image content as the response. (We can stream the file we just saved or serve from memory. A simple way: after saving, do return FileResponse(path=that_path, media_type="image/png").) - Alternatively, we could avoid saving to disk first and stream the PIL image directly (as seen in the earlier example[3]). However, since we want it logged and saved anyway, writing to disk then reading is fine. Or we do both: save to disk for log, and also send the in-memory bytes to the user to save a disk read roundtrip. Implementation detail: e.g.
buf = io.BytesIO()
image.save(buf, format="PNG")
buf.seek(0)
return StreamingResponse(buf, media_type="image/png")
– and concurrently save to file. This avoids an extra disk read for the response.
Note: For simplicity, using FileResponse after saving may be easiest – it will use efficient file handling under the hood.
Example:
GET /api/p/82f6qk/generate?prompt=Why did the chicken cross the road?
- If "Why did the chicken cross the road?" is a configured trigger in prank set 82f6qk, the response will be the exact image the creator uploaded for that prompt (e.g. a meme image). The user will see that image. - If the prompt was something else (not matching any trigger), e.g. "A beautiful landscape painting", the backend might route to flux_dev (for example) and generate a new image, then return that image. The user sees a normal AI-generated result, not suspecting that certain secret prompts would have produced different outcomes.
This dual behavior is the crux of the prank feature.
5. GET /api/prank-sets/{creator_slug} – Get prank set details and prompts
Description: Retrieves the list of trigger prompts and their images for a given prank set. This is used in the prank customization interface to show the user all the triggers they’ve set up, along with thumbnails, and possibly to allow deletion or editing.
    • URL Path: {creator_slug} – the private slug of the prank set (to ensure only someone with the slug can view the details).
    • Response: JSON object with details:
    • prank_set_id, creator_slug, share_slug, title, created_at of the prank set (could be top-level or inside a prank_set object).
    • prompts: an array of trigger entries, each with:
        ◦ id (prompt id),
        ◦ trigger_prompt (text),
        ◦ thumbnail_url (small image preview),
        ◦ created_at.
We likely do not send full image URLs here to save bandwidth – thumbnails are enough for listing. If the user wants to see the full image they uploaded, the UI could use the thumbnail link or have a separate endpoint to fetch the full image (or just change the URL to the non-thumb image path, since it follows a pattern).
Logic: 1. Look up the prank set by creator_slug. If not found, 404. 2. Retrieve all prank_prompts for prank_set_id (ordered by created_at maybe). 3. Construct the response JSON with the prank set info and an array of prompts:
{
  "prank_set_id": 12,
  "creator_slug": "9xg2kt",
  "share_slug": "82f6qk",
  "title": "My Prank Set",
  "created_at": "...",
  "prompts": [
      {
         "id": 34,
         "trigger_prompt": "Why did the chicken cross the road?",
         "thumbnail_url": "/images/pranks/prank_34_thumb.jpg",
         "created_at": "2025-11-21T18:30:45Z"
      },
      {
         "id": 35,
         "trigger_prompt": "Hello there",
         "thumbnail_url": "/images/pranks/prank_35_thumb.jpg",
         "created_at": "2025-11-21T18:32:10Z"
      }
  ]
}
4. Return that JSON. The frontend can use this to render a list of current triggers with thumbnails.
6. DELETE /api/prank-sets/{creator_slug}/prompts/{prompt_id} – Remove a trigger from a prank set
Description: Allows the prank set creator to delete one of their trigger prompt-image pairs. This might be called when the user clicks a “delete” button on a trigger in the UI.
    • URL Path: creator_slug for the set, and the specific prompt_id to delete.
    • Response: likely just a confirmation JSON or status code 204 No Content on success.
Logic: 1. Verify the prank set by creator_slug exists. (We could also verify that the prompt_id corresponds to that set, to avoid accidentally deleting a prompt from another set.) - E.g. DELETE /api/prank-sets/9xg2kt/prompts/35: look up prank_prompts where id=35 and get prank_set_id, ensure that prank_set_id matches the prank_set with creator_slug=9xg2kt. - If not matching or not found, return 404 or 403. 2. Delete the prank_prompts record (SQL DELETE). 3. Delete the image file and thumbnail file from disk (to free space). Use the paths stored in the DB for that prompt to locate the files and remove them.
import os
os.remove(os.path.join("data/images", <image_path>))
os.remove(os.path.join("data/images", <thumbnail_path>))
(Wrap in try/except in case file already missing, etc.) 4. Return success (could be {"detail":"Deleted"} or just HTTP 204).
7. PUT /api/generation/{generation_id}/edit – Upload an edited image for a generation
Description: Handles the case where a user has taken an AI-generated image and manually edited it (e.g. in a drawing interface) and now wants to save this edited version. We treat it as a new generation entry that links to the original.
    • URL Path: generation_id – the ID of the original generation that was edited.
    • Request: form-data (since an image file is included):
    • image – the edited image file (e.g. PNG or JPEG).
    • (Optionally, if any metadata about the edit, but likely just the image itself.)
    • Response: JSON with the new generation entry:
    • generation_id: new ID for this edited version.
    • prompt: (same as original prompt, carried over).
    • model_id: (same model as original, since this is an edit of that output).
    • image_url: path to the edited image.
    • thumbnail_url: path to the edited thumbnail.
    • version_number: e.g. 2 (if original was 1).
    • edited_from_generation_id: the original generation_id (link back).
    • created_at: timestamp.
Logic: 1. Find the original generation log by generation_id. If not found, 404. 2. Receive the uploaded edited image file. Load it with PIL if we need to process, or we can directly save the bytes. Likely we want to standardize format, so open with PIL:
edited_image = Image.open(upload.file)
if edited_image.mode in ("RGBA", "P"):
    edited_image = edited_image.convert("RGB")
3. Determine new file names. We can either: - Use a similar naming scheme but indicate version. For instance, if original image_path was generations/gen_57.png, we could name this generations/gen_57_v2.png. However, that could conflict if multiple separate edits branch off the same original. Instead, treating every generation (even edited ones) as a new entry with its own ID is simpler. - So use the new generation ID or a UUID. We could incorporate original id in name for clarity, e.g. gen_57_edit_1.png, but let's stick to ID or UUID. - We won't know new ID until after DB insert unless we prefetch. We could do an insert to get ID first (since this operation is quick relative to image gen). 4. Insert the new generation log entry: - prompt = original prompt, - model_id = original model_id (we keep it for reference), - router_json = NULL or we could copy original’s router_json (since the model choice reasoning is same as original). Probably we set it NULL or something like {"reason": "manual edit"}, - image_path and thumbnail_path to be determined, - prank_set_id, creator_slug, share_slug are copied from original if they existed (if the original was from a share link, we might carry over that context; if it was a normal generation, these are null), - version_number = original.version_number + 1 (if original was v1, new is v2), - edited_from_generation_id = original.id, - created_at = now(). - If using SQLAlchemy, we can create the object without image_path yet, flush to get an ID. 5. Now save the edited image to file. If we got the new ID (say 58), we can name file gen_58.png.
new_path = f"data/images/generations/gen_{new_id}.png"
edited_image.save(new_path, format="PNG")
Make thumbnail similarly and save as gen_{new_id}_thumb.jpg. 6. Update the generation_logs entry for new_id with the image_path and thumbnail_path. 7. Return the new entry JSON (similar to generation response).
Example: Suppose generation 57 (version 1, model flux_dev) was edited. Request:
PUT /api/generation/57/edit
Form: image=@edited_image.png
Response:
{
  "generation_id": 58,
  "prompt": "A fantasy castle on a hill during sunset, digital art",
  "model_id": "flux_dev",
  "image_url": "/images/generations/gen_58.png",
  "thumbnail_url": "/images/generations/gen_58_thumb.jpg",
  "version_number": 2,
  "edited_from_generation_id": 57,
  "created_at": "2025-11-21T18:45:00Z"
}
(If the user continues editing further, each PUT to edit will create another entry, potentially forming a chain: 57 -> 58 -> 59 etc, with version 3,4,... and edited_from_generation_id always pointing to the immediate previous one or the original – we chose immediate previous to allow branching, but one could also always reference the original. Here we assume sequential editing.)
8. GET /api/generation-logs – Retrieve the history of generations
Description: Provides a list of generation logs. This could be used for an admin interface or to show users recent images (if multi-user, one would filter by user; but here no login, so it’s global or perhaps just for debugging). We will implement it as a simple log listing with optional query params for pagination.
    • Query Params (optional):
    • limit (int, default 50): number of records to return.
    • offset (int, default 0): for pagination start.
    • We could also allow filtering by share_slug or creator_slug for context, but not specified.
    • Response: JSON with an array of log entries. Each entry could include:
    • id, prompt, model_id, image_url, thumbnail_url, created_at, version_number.
    • Possibly share_slug or prank_set_id if present (to know context).
    • Possibly the router_json or at least the tags/reason from it for insight.
Because router_json can be large, we might not include full details by default. Maybe extract key info (like selected model and tags).
For completeness, we can include it as a nested object or string.
Logic: 1. Query generation_logs ordered by created_at DESC (most recent first), with the given limit/offset. 2. For each entry, build a dictionary with the fields. Construct the image_url and thumbnail_url from stored paths (prefix with /images/). 3. Return the list (or an object with logs: [...]).
Example:
GET /api/generation-logs?limit=2
Response:
{
  "logs": [
    {
      "id": 58,
      "prompt": "A fantasy castle on a hill during sunset, digital art",
      "model_id": "flux_dev",
      "image_url": "/images/generations/gen_58.png",
      "thumbnail_url": "/images/generations/gen_58_thumb.jpg",
      "version_number": 2,
      "edited_from": 57,
      "share_slug": null,
      "created_at": "2025-11-21T18:45:00Z",
      "router_decision": {
          "model_id": "flux_dev",
          "tags": ["landscape"],
          "reason": "No specific indicators for other models."
      }
    },
    {
      "id": 57,
      "prompt": "A fantasy castle on a hill during sunset, digital art",
      "model_id": "flux_dev",
      "image_url": "/images/generations/gen_57.png",
      "thumbnail_url": "/images/generations/gen_57_thumb.jpg",
      "version_number": 1,
      "edited_from": null,
      "share_slug": null,
      "created_at": "2025-11-21T18:37:12Z",
      "router_decision": {
          "model_id": "flux_dev",
          "tags": ["landscape"],
          "reason": "No specific indicators for other models."
      }
    }
  ]
}
(The router_decision here is a pared-down version of the full JSON, just showing some key fields; we could include the whole JSON as well.)
Additional Implementation Notes
    • FastAPI App Structure: We can organize the code with routers or all endpoints in a single main.py. For clarity, consider splitting into modules:
    • models.py – SQLAlchemy models and maybe Pydantic schemas.
    • router_engine.py – code to interface with the LLM and implement route_model(prompt, prank_triggers=None).
    • generate.py – endpoints related to generation and editing.
    • prank.py – endpoints for prank set management.
    • Then include these routers in the main FastAPI app.
    • Model Inference Performance: Running the router LLM for every generation request adds overhead. Llama 8B on CPU might take a couple of seconds to respond. This is likely acceptable given image generation itself might take several seconds. For faster checks (like very trivial prompt matching), one could use an embedding approach or direct string similarity as a quick filter before calling the LLM for fine judgment. For example, if a user prompt exactly matches a prank trigger or is very close (we could do lowercase and strip and compare), we could short-circuit and consider it a match without the LLM. But the LLM approach covers semantic similarity (like rephrased prompts).
    • Error Handling: We should handle possible errors:
    • If model generation fails (e.g. out of VRAM or other issues), catch exceptions and return HTTP 500 with an error message.
    • If the router LLM fails to produce valid JSON, we might have it try again or default to flux_dev. Also log such cases for debugging.
    • In all file operations (saving images), handle I/O errors (disk full, etc.) – though unlikely.
    • Security Considerations: Since there's no auth, the only thing protecting prank set editing is the secrecy of the creator_slug. We must ensure that endpoints that modify a prank set (adding prompts, deleting) are only accessible via the creator slug path. We should not expose any listing of all prank sets or a way to derive the slug from the share slug in the API. Also, using long random slugs mitigates brute force guessing (36^6 ~ 2.1 billion combos). If more security is needed, could extend slug length to 8+ chars.
    • Privacy: All generated images and uploads are stored on disk. The share link images are accessible by knowing their URLs. We might keep filenames obscure (using UUIDs) to prevent someone from incrementing an ID in URL to fetch another’s image. In our scheme, generation images are somewhat guessable by ID if one is inclined (since gen_57.png, gen_58.png, etc. are sequential). If this is a concern, using UUID names or including part of the slug in the filename could obscure it. For instance, if generation was done via share slug, we could incorporate share slug in file name. But given this is a niche app and no personal data, it might be acceptable.
    • Thumbnail Format: We chose JPEG for thumbnails for compatibility and compression. Since some generated images might have transparency (PNG with alpha), we converted to RGB for JPEG[8], which means thumbnail background will be black if original had transparency. Alternatively, we could use WebP for thumbnails to support alpha (or just stick to JPEG and not worry about transparent images much, as AI outputs are usually full images).
    • Background Tasks: As noted, FastAPI’s BackgroundTasks could be used to do things like thumbnail generation or log writing after responding to the request. For example, if latency is critical, we could respond with the image as soon as it’s generated and offload the disk saving and database logging to a background task. However, in this design, since we want a consistent log and need the ID, it’s simpler to finish all work before responding. If high throughput is needed, more sophisticated queuing would be implemented.
    • Fine-tuning the Router: The design mentions that later on, the router LLM can be fine-tuned using the data in generation_logs. This suggests we might collect prompts, chosen models, and outcomes to improve the router’s performance over time (so it learns the patterns of which model was actually good for which prompt). Implementing that is outside the scope of the backend server itself, but we ensure to log everything needed (prompt, model_id, maybe user feedback if any). Fine-tuning would be an offline process resulting in a new version of the router model.
    • Example Code Snippet (FastAPI Endpoint): To tie it together, here’s a simplified example of what the POST /api/generate endpoint implementation might look like using FastAPI:
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import json

app = FastAPI()

# Assume models are loaded and executor is set up on startup
# ...

class GenerateRequest(BaseModel):
    prompt: str
    engine: str = "auto"

@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    prompt = req.prompt
    engine = req.engine
    # Route or select model
    router_meta = None
    if engine == "auto":
        decision = route_model(prompt)  # our function that uses LLM
        model_id = decision["model_id"]
        router_meta = decision  # we'll include it in response
    else:
        model_id = engine
    # Run generation
    try:
        result = executor.submit(model_pipes[model_id], prompt).result()
        pil_image = result.images[0]
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Generation failed: {e}"})
    # Save image and thumbnail
    gen_id = create_generation_log_placeholder(prompt, model_id, router_meta)  # insert to DB to get ID
    img_path = f"generations/gen_{gen_id}.png"
    thumb_path = f"generations/gen_{gen_id}_thumb.jpg"
    pil_image.save(os.path.join("data/images", img_path), format="PNG")
    # create thumbnail
    thumb = pil_image.copy()
    thumb.thumbnail((256,256))
    thumb.save(os.path.join("data/images", thumb_path), format="JPEG", quality=80)
    # update DB entry with paths, etc.
    finalize_generation_log(gen_id, img_path, thumb_path)
    # Build response
    resp = {
        "generation_id": gen_id,
        "model_id": model_id,
        "image_url": f"/images/{img_path}",
        "thumbnail_url": f"/images/{thumb_path}",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    if router_meta:
        resp["router_metadata"] = router_meta
    return resp
(Functions like route_model, create_generation_log_placeholder, and finalize_generation_log represent the logic we've described: calling the LLM, inserting into DB, updating DB. This is just to illustrate the flow.)
Likewise, other endpoints would follow the logic described above.
By following this plan, a developer or AI coding agent (Copilot/Codex) could implement a robust backend that meets all the requirements: handling multiple models with intelligent routing[9], supporting user-contributed prank outputs, managing data persistence, and providing clear APIs for the front-end integration. Each component (from slug generation[4] to image handling[7] to model inference[1][3] to LLM usage) has a defined role in the system, ensuring the end result is a seamless and fun experience for both the prank creators and the unsuspecting users of the share links.

[1] [3] [5] Stable Diffusion Inference using FastAPI and load testing using Locust - DEV Community
https://dev.to/dhruvthu/stable-diffusion-inference-using-fastapi-and-load-testing-using-locust-41pc
[2] [9]  Using Llama in FastAPI - Abdulhamit Celik 
https://www.abdulhamitcelik.com/en/blog/fastapi-llama-implementation/
[4] python - Random string generation with upper case letters and digits - Stack Overflow
https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
[6] Optimization of images with FastAPI | by Nelson Hernández | Medium
https://nelsoncode.medium.com/optimization-of-images-with-fastapi-2a1427b57358
[7] [8] How to save an uploaded image to FastAPI using Python Imaging Library (PIL)? - Stack Overflow
https://stackoverflow.com/questions/73810377/how-to-save-an-uploaded-image-to-fastapi-using-python-imaging-library-pil