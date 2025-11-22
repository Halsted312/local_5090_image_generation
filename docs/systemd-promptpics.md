# Systemd setup to auto-start PromptPics (docker + ngrok)

This assumes your project lives at `/home/halsted/Python/flexy-face`, Docker is installed, and your ngrok config is in `~/.config/ngrok/ngrok.yml` with the `promptpics` endpoint.

## 1) Service files

Save these under `/etc/systemd/system/` as root:

`/etc/systemd/system/promptpics-docker.service`
```
[Unit]
Description=PromptPics docker compose stack
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=oneshot
WorkingDirectory=/home/halsted/Python/flexy-face
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
RemainAfterExit=yes
TimeoutStartSec=0
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/promptpics-ngrok.service`
```
[Unit]
Description=Ngrok tunnel for PromptPics
After=network-online.target promptpics-docker.service
Wants=promptpics-docker.service

[Service]
Type=simple
ExecStart=/home/halsted/Python/flexy-face/scripts/start_ngrok_promptpics.sh
Restart=on-failure
User=halsted
WorkingDirectory=/home/halsted/Python/flexy-face
Environment=NGROK_CONFIG=/home/halsted/.config/ngrok/ngrok.yml

[Install]
WantedBy=multi-user.target
```

## 2) Enable + start
```
sudo systemctl daemon-reload
sudo systemctl enable promptpics-docker.service promptpics-ngrok.service
sudo systemctl start promptpics-docker.service promptpics-ngrok.service
```

## 3) Check status/logs
```
systemctl status promptpics-docker.service
journalctl -u promptpics-docker.service -f
systemctl status promptpics-ngrok.service
journalctl -u promptpics-ngrok.service -f
```

## 4) Manual scripts (if you prefer)
- Start stack: `cd /home/halsted/Python/flexy-face && docker compose up -d`
- Start ngrok: `./scripts/start_ngrok_promptpics.sh`
- Stop stack: `docker compose down`
