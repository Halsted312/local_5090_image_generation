Here’s a full “spec” you can hand to your coding agents (desktop + Replit) that covers:

* **Tech stack** (modern, animation‑friendly, responsive)
* **How it talks to your GPU backend**
* **Exact layout & UX for the landing page + `/imagine` super mode**
* **Menu items and what each section should say**

I’ll frame it as instructions you can almost copy‑paste.

---

## 1. Front‑end tech stack (Replit / promptpics.ai)

Ask the Replit agent to build the front‑end as a **React SPA with Vite + Tailwind + Motion + TanStack Query**:

**Core stack**

* **React 18+** – standard UI library. React docs recommend using a build tool like Vite for modern apps.
* **Vite** – dev server + build tool (super fast, great DX vs CRA/Webpack).
* **Tailwind CSS** – utility‑first CSS for fast, responsive UI without writing tons of custom CSS.
* **Motion (Framer Motion)** – animation library for React (hover effects, page transitions, scroll animations).
* **TanStack Query (React Query)** – handles API calls, caching, loading states, retries.

**What to tell the Replit agent technically**

> * Create a new **Vite + React + TypeScript** project.
> * Install:
>
>   * `tailwindcss postcss autoprefixer` (Tailwind)
>   * `@tanstack/react-query` (TanStack Query)
>   * `motion` (Motion/Framer Motion for React)
> * Configure Tailwind + PostCSS according to the Tailwind docs (generate `tailwind.config.cjs`, add Tailwind directives to `src/index.css`).
> * Wrap the app in a `QueryClientProvider` from TanStack Query for data fetching state.
> * Use Tailwind for layout and responsive design, Motion for animations (hero, cards, section reveals, button hover effects).

---

## 2. Backend integration: how the SPA talks to your GPU desktop

Your desktop already runs the FastAPI backend, proxied by ngrok at:

> **`https://app.promptpics.ai` → FastAPI on localhost:7999**

We’ll treat this as a **pure API**.

**Instructions for your desktop/VS Code agent (you partly did this already):**

> * Make sure ngrok config (`~/.config/ngrok/ngrok.yml`) has:
>
>   ```yaml
>   endpoints:
>     - name: promptpics
>       url: https://app.promptpics.ai
>       upstream:
>         url: 7999
>   ```
> * In FastAPI, configure CORS:
>
>   ```python
>   origins = [
>       "https://promptpics.ai",
>       "https://*.replit.app",
>       "http://localhost:3000",
>   ]
>   app.add_middleware(
>       CORSMiddleware,
>       allow_origins=origins,
>       allow_credentials=False,
>       allow_methods=["*"],
>       allow_headers=["*"],
>   )
>   ```
> * Keep current APIs:
>
>   * `POST /api/generate`
>   * `POST /api/edit`
>   * `POST /api/pranks`
>   * `POST /api/pranks/{prank_id}/triggers`
>   * `POST /api/p/{slug}/generate`

**Instructions for Replit agent (API client):**

> * Use env var `VITE_API_BASE_URL` = `https://app.promptpics.ai`.
> * Create `src/api.ts` that exports:
>
>   * `generateImage(payload)` → `POST ${API_BASE}/api/generate`
>   * `editImage(file, prompt, steps)` → `POST ${API_BASE}/api/edit` (FormData)
>   * `createPrank(title)` → `POST ${API_BASE}/api/pranks`
>   * `addPrankTrigger(prankId, text, file)` → `POST ${API_BASE}/api/pranks/{id}/triggers`
>   * `generatePrank(slug, prompt)` → `POST ${API_BASE}/api/p/${slug}/generate`
> * Wrap each in TanStack Query mutations/queries to handle loading, error states, caching.

---

## 3. Global layout & navigation

**Top‑level nav (desktop view)**

Fixed top nav bar with:

* **Logo**: “PromptPics” (simple text logo initially; can become a logotype later).
* Menu items (scroll or route to sections):

  * **Home** (`/` – scroll to hero)
  * **How It Works**
  * **Super Mode / Imagine** (link to `/imagine`)
  * **Tech** (Technical section)
  * **About**
  * **Contact**

On mobile:

* Collapse into hamburger menu with a slide‑down Motion animation.

**Instructions to Replit agent (nav UI + animation):**

> * Use a sticky `<header>` with a translucent background and blur (`backdrop-blur`) when scrolled.
> * On desktop, show inline nav links; on mobile, use a hamburger button that toggles a Motion animated drop‑down menu.
> * Use Motion for:
>
>   * Fade/slide in of the nav on load.
>   * Smooth underline/scale animations on hover for menu items.

---

## 4. Home page: section‑by‑section layout & copy

### 4.1. Hero section (above the fold)

**Goals:**

* Communicate what PromptPics is in 1–2 lines.
* Show the chat + image generator immediately (or at least a preview).
* Strong primary action.

**Layout:**

* Two‑column on desktop, stacked on mobile:

  * Left: text + primary CTA.
  * Right: live preview card or the actual mini generator.

**Copy suggestions:**

* Title: **“PromptPics”**
* Subtitle: **“Turn any idea into an image in seconds.”**
* Sub‑sub: “Type a prompt. Get AI images. Share prank links that secretly show your own photos.”
* Buttons:

  * **[Try it now]** → scrolls to the chat/generator section.
  * Secondary: **[Super Mode]** → `/imagine`.

**Hero animation ideas:**

* Background gradient animated subtly with Motion.
* The mock “image card” on the right:

  * Animated in with a slight scale/opacity.
  * Cycle through 2–3 placeholder prompts/images every few seconds with a Motion or CSS keyframe crossfade.
* CTA buttons:

  * Hover scale + drop shadow with Motion.

---

### 4.2. Chat + Image Generator section (“Live Demo”)

This is the core of page `/` just below hero.

**Layout:**

* Responsive 2‑column (`md:grid-cols-2` in Tailwind):

  * Left: chat prompt area + controls.
  * Right: image output + history grid.

**UI elements:**

* Chat box:

  * A “message bubble” listing last 3 prompts the user entered.
  * At bottom: single input + “Generate” button.

* Controls:

  * Two small labeled sliders:

    * “Quality (steps)” → maps to `num_inference_steps`.
    * “Creativity (guidance)” → maps to `guidance_scale`.

* When generating:

  * Disable input and show a **progress bar or timer** (“~3s remaining”) – even if it’s approximate, it makes the UX feel responsive.
  * Use Motion to animate a skeleton placeholder (pulsing gradient rectangle) where the image will appear.

* Output panel:

  * The latest generated image as a large aspect‑ratio card.
  * Below, a horizontal scroll of previous images with their prompts.
  * Hovering an old thumbnail fades its prompt in overlay text.

**Technical instructions:**

> * Wrap all generate/edit calls in TanStack Query `useMutation` hooks so you get isLoading / isError states.
> * While `isLoading`:
>
>   * Show animated skeleton using Tailwind gradients + Motion.
>   * Disable the “Generate” button and show a spinner icon with `animate-spin`.
>   * Optionally, show a simple `setInterval`‑driven timer counting seconds until completion.
> * On success:
>
>   * Push the new `{prompt, imageBase64}` into a React state array and render as cards.
>   * Smoothly animate in the new card from opacity 0 / translateY(8px) to visible with Motion.

---

### 4.3. “How It Works” section

Anchor: `#how-it-works`. Nav link **How It Works** scrolls here.

**Visual:**

* 3 or 4 cards in a horizontal row on desktop (stacked on mobile), each with an icon, title, description.

**Step suggestions:**

1. **Type a prompt**
   “Describe anything you can imagine – a scene, a style, or a vibe.”
2. **Our models go to work**
   “We use modern text‑to‑image models running on a GPU rig to turn words into pictures.”
3. **Download or tweak**
   “Download your result, regenerate, or lightly edit it with a new instruction.”
4. **Optional: Super Mode (prank links)**
   “Create special links that act like a normal AI generator – except for prompts you choose.”

**Technical detail snippet inside this section:**

Short block explaining that prompts are sent from the browser to a backend API running on GPUs:

> “When you click Generate, your browser sends your prompt to our backend API. We run high‑quality image models on a dedicated GPU machine and stream the image back to your device.”

Use Motion to stagger the cards in with a fade/slide animation as they scroll into view.

---

### 4.4. “Super Mode” / `/imagine` teaser section

Anchor: `#super-mode`. Link in nav: **Super Mode**.

**Layout:**

* Half section, with:

  * Left: explanation text.
  * Right: a mock UI preview of the `/imagine` page.

**Copy idea:**

* Title: “Super Mode: Create prank links that feel real”
* Text:
  “Design custom prompts and upload your own images. Share a link that behaves like a normal AI generator for most prompts – but shows your secret image when someone types what you expect.”

Button: **[Open Super Mode]** → navigate to `/imagine`.

---

### 4.5. “Tech” / Technical section

Anchor: `#tech`. Nav: **Tech**.

Purpose: sound smart and honest about your stack.

**Layout:**

* Two‑column on desktop:

  * Left: summary bullet points.
  * Right: a simple diagram or code snippet.

**Copy:**

> **Under the hood**
>
> * **Frontend:** React SPA built with Vite, styled with Tailwind CSS for clean, responsive layouts.
> * **Animations:** Motion (Framer Motion) for smooth scroll‑triggered effects, micro‑interactions, and transitions.
> * **Backend API:** Python / FastAPI running on a GPU‑equipped machine, exposed securely via ngrok.
> * **Data layer:** TanStack Query in the front‑end manages API calls, caching, and loading states to keep the UI snappy.
> * **Models:** Text‑to‑image diffusion models (like FLUX) with custom logic to compare prompts and route to real/prank images.

You can add a tiny “Architectural diagram” (even ASCII or SVG) showing:

`Browser (promptpics.ai) → app.promptpics.ai (FastAPI) → GPU image models`

---

### 4.6. “About” section

Anchor: `#about`. Nav: **About**.

**Content:**

* Small founder story (“Built by Stephen, a data scientist who loves generative art & pranks”).
* Why you built it (fun + experimentation with GPUs + LLM matching).
* Emphasize privacy: images are only used to fulfill the request, no public gallery unless explicitly added later.

---

### 4.7. “Contact” section

Anchor: `#contact`. Nav: **Contact**.

**Layout:**

* Minimal contact form (name, email, message) – this can initially just send to an email service or be a stub.
* Alternatively, a simple “Coming soon – contact me at [email] or [X handle]”.

Add subtle Motion on focus/hover of inputs and submit button.

---

## 5. `/imagine` page (Super Mode admin UI) layout & behavior

This is a dedicated route in the SPA that uses the same header + footer, but main content is a 2‑column builder.

**Layout:**

* **Top:** Title + warning

  * “Super Mode: Prank Link Builder”
  * Small text: “This page lets you create prank links. Don’t use it to harass or harm people.”

* **Body:** 2 columns on desktop, stacked on mobile.

  **Left side: Prank configuration**

  * Input: “Prank title” (optional, stored with prank).
  * Button: “Create prank” → calls `createPrank`.

    * On success, show generated `slug` and full link.
  * After prank exists:

    * A list of trigger rows:

      * Text input / textarea: “Trigger prompt”
      * File upload
      * “Add trigger” button → calls `addPrankTrigger`.
    * Display added triggers in a mini table: [Prompt text, thumbnail, created time].

  **Right side: Preview & share**

  * Show “Share link”: `https://promptpics.ai/p/<slug>` or `https://app.promptpics.ai/p/<slug>`.
  * Copy button (copy to clipboard) with a Motion ripple.
  * Simple test area:

    * Text box “Test prompt”.
    * Button “Test as friend” → calls `generatePrank(slug, prompt)` and shows which image came back (prank vs AI).

**Analytics you can track:**

* `super_mode_open`
* `prank_created`
* `prank_trigger_added`
* `prank_link_copied`
* `prank_tested`

---

## 6. UX details and animation patterns (what to emphasize)

Ask the Replit agent explicitly for:

* **Responsive design** using Tailwind’s breakpoints (e.g. `sm:`, `md:`, `lg:`) so hero and generator layout collapse gracefully on phones.
* **Animations with Motion:**

  * Page sections fading/sliding in when they enter viewport.
  * Buttons with hover scale + shadow.
  * Image cards animating in on generate.
  * Smooth transitions between `/` and `/imagine` routes.
* **Loading & timers:**

  * Use TanStack Query’s `isLoading` + `isFetching` to control skeletons and spinners.
  * Simple timer (using `useEffect` + `setInterval`) during image generation to show “X seconds so far”.

This combo gives you the “latest tech feel”: fast dev/build (Vite), clean responsive UI (Tailwind), smooth animations (Motion), and robust API behavior (TanStack Query).

---

If you’d like, next step I can turn all of this into a single “prompt” for your Replit agent, written as instructions they can follow line‑by‑line (including component names, file structure, and a starter Tailwind design system).
