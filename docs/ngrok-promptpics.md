# Exposing `app.promptpics.ai` via ngrok

Use a reserved ngrok custom domain to tunnel your local stack. These steps assume:

- Frontend (Nginx + built React) available on `http://127.0.0.1:7080`.
- Backend FastAPI runs on `http://127.0.0.1:7999` and is proxied via Nginx at `/api`.
- Domain managed in GoDaddy: `promptpics.ai`.
- Reserved subdomain in ngrok: `app.promptpics.ai`.

## 1) Reserve the domain in ngrok
1. In the ngrok dashboard, go to **Domains → New Domain**.
2. Enter `app.promptpics.ai`.
3. Copy the generated **CNAME target** (e.g. `xxxxx.ngrok-cname.com`).

## 2) Add CNAME in GoDaddy
1. Go to **My Products → DNS / Manage DNS** for `promptpics.ai`.
2. Add a **CNAME** record:
   - Host/Name: `app`
   - Value/Points to: the ngrok CNAME target.
3. Save. Return to ngrok and verify the domain once DNS propagates.

## 3) Local ngrok config (v3)
Create or edit `~/.config/ngrok/ngrok.yml`:

```yaml
version: 3

agent:
  authtoken: <YOUR_NGROK_AUTHTOKEN>

endpoints:
  - name: promptpics
    url: https://app.promptpics.ai   # public domain
    upstream:
      url: 127.0.0.1:7080            # Frontend on host (Nginx), proxies /api to backend
```

## 4) Run the tunnel

Ensure the stack is running (frontend on 7080, backend on 7999 behind Nginx), then:

```bash
./scripts/start_ngrok_promptpics.sh
```

This runs `ngrok start promptpics` using the config above.

## 5) Notes
- CORS is preconfigured to allow `https://app.promptpics.ai`.
- If you change the backend port, update the `upstream.url` and the script.
- To auto-run at boot, you can create a systemd unit that calls `ngrok start promptpics`.
