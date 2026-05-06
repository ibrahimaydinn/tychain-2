# 🚀 Deploy Tychain (Production)

This is the up-to-date deploy guide for the merged Tychain stack:
**Flask + Gunicorn → Docker → Hugging Face Spaces**, with **Turso (libSQL)**
as the persistent database and SMTP for outbound alert emails.

> ⚠️ **Never commit secrets.** All credentials live in the Space's
> "Variables and secrets" panel and are read at runtime via `os.environ`.

---

## 0. Prerequisites

- A GitHub account with a repo (this guide assumes `tychain-1`).
- A Turso account (`turso auth login`) with a created database.
- A Hugging Face account.
- An SMTP provider (Gmail App Password, AWS SES, Mailgun, …) — optional;
  leave `SMTP_HOST` unset to keep alerts in mock-log mode.

---

## 1. Create / verify the Turso database

```bash
turso db create tychain-1                       # if not already created
turso db show tychain-1 --url                   # → libsql://tychain-1-<org>.turso.io
turso db tokens create tychain-1                # → JWT auth token (keep secret)
```

Tables are created automatically on first app boot via `init_db()`. To
verify the schema:

```bash
TURSO_DATABASE_URL=libsql://… TURSO_AUTH_TOKEN=… python db_config.py
# Should print "Test query result = [(1,)]" — connection works.

TURSO_DATABASE_URL=libsql://… TURSO_AUTH_TOKEN=… \
  python -c "import app; app.init_db(); print('schema OK')"
```

---

## 2. Push the code to GitHub

The repo expects to live at `https://github.com/<you>/tychain-1`.
Use SSH or `gh` CLI auth — **don't** paste a PAT into a chat or commit it.

```bash
cd sp500-bist30-1

# First-time setup
git init -b main
git add .
git commit -m "Initial deploy: Tychain (Flask + Turso + HF Spaces)"

git remote add origin https://github.com/<you>/tychain-1.git
git push -u origin main
```

If pushing over HTTPS, authenticate via either:

- `gh auth login` (recommended), or
- A credential helper: `git config --global credential.helper osxkeychain`
  (macOS) / `manager` (Windows) / `cache` (Linux).

---

## 3. Create the Hugging Face Space

1. Go to **https://huggingface.co/new-space**
2. **Owner:** your account · **Space name:** `tychain`
3. **License:** MIT · **SDK:** **Docker** (NOT Gradio — we ship our own
   `Dockerfile`) · **Hardware:** CPU Basic (free)
4. **Public** (so others can visit) · click **Create Space**

### 3a. Wire the Space to your GitHub repo

In **Settings → Repository secrets / Sync**, either:

- **Push from GitHub** (preferred) — add the Space as a second remote:
  ```bash
  git remote add space https://huggingface.co/spaces/<you>/tychain
  git push space main:main
  ```
- Or **mirror** the GitHub repo via the HF UI ("Sync with GitHub").

The Space rebuilds automatically on every push.

### 3b. Set Secrets

Open **Settings → Variables and secrets → New secret** and add:

| Name | Value |
|---|---|
| `TURSO_DATABASE_URL` | `libsql://tychain-1-<org>.turso.io` |
| `TURSO_AUTH_TOKEN` | your Turso JWT |
| `SECRET_KEY` | `python -c "import secrets;print(secrets.token_hex(32))"` |
| `APP_URL` | `https://huggingface.co/spaces/<you>/tychain` |
| `SMTP_HOST` | e.g. `smtp.gmail.com` *(omit to disable real email)* |
| `SMTP_PORT` | `465` |
| `SMTP_USER` | your SMTP login |
| `SMTP_PASS` | SMTP password / app-password |
| `MAIL_FROM` | `alerts@yourdomain.tld` |
| `MAIL_NAME` | `Tychain Alerts` |

Secrets are exposed to the container as environment variables — `app.py`
and `db_config.py` already read them via `os.environ`.

---

## 4. First boot

The Docker image will:
1. Install `requirements.txt` (≈ 4–6 min, mostly TensorFlow + libSQL wheel).
2. Boot Gunicorn on port 7860.
3. Run `init_db()` → creates tables on Turso if missing.

When the Build tab turns green, visit the Space URL and create an account.

---

## 5. Operations

| Task | Command |
|---|---|
| Tail Space logs | HF Space → **Logs** tab |
| Force a rebuild | HF Space → **Settings → Factory reset** |
| Run a signal scan manually | `POST /api/tracked` with `action=cron` (admin) |
| Rotate Turso token | `turso db tokens invalidate tychain-1 && turso db tokens create tychain-1` then update HF Secret |
| Inspect DB | `turso db shell tychain-1` |

---

## 6. Local development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then fill in TURSO_* (or leave unset for SQLite)
python app.py              # http://127.0.0.1:8080
```

Without `TURSO_DATABASE_URL`, the app uses local `tychain.db` so you can
develop offline.

---

## 7. Troubleshooting

- **`libsql-experimental` build fails** → confirm Python ≥ 3.11 (the
  Dockerfile uses `python:3.11-slim` for this reason).
- **`TURSO_AUTH_TOKEN missing`** → set it as a *secret*, not a *variable*.
- **Login attempts fail with `IntegrityError`** → unique-email collision;
  expected behaviour.
- **Emails not arriving** → confirm `SMTP_HOST` is set; check Space logs
  for `[email_helper] Failed to send …`.
- **Space sleeps** → upgrade hardware in Settings, or hit it with a
  cron-job.org keep-alive ping every 25 min.
