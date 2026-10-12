# Hosting Gurasees's Deep Learning Local LLM Chatbot online

## What can run where

The chatbot has two parts:

| Part | What it is | Can run on Netlify / Vercel / Cloudflare Pages? |
|---|---|---|
| **Website** (`gpt-oss-chat/static`) | The chat page: HTML + JavaScript | **Yes**, it is a static site |
| **Chat server** (`gpt-oss-chat/server.py`) | Python server that talks to **Ollama**, reads web links, makes ZIPs, runs the image worker | **No.** Those hosts can't run Ollama or keep a Python server and a 4 GB+ GPU model alive |

So the models always run on a computer you control: **your PC** (free) or a rented server with a GPU (paid).
Netlify / Vercel / Cloudflare Pages can only host the *website* part, and it then talks to the chat server over the internet.

## Option A: share straight from your PC (free, simplest, recommended)

The tunnel serves the website *and* the server together, so there is nothing else to deploy.

1. Install the tunnel program once: `winget install Cloudflare.cloudflared`
2. Double-click **`gpt-oss-chat/start_public.bat`**, choose a password.
3. Copy the `https://….trycloudflare.com` address that appears in the tunnel window and share it with the password.

Notes:
- Your PC must stay on, and the site goes offline when you close the window.
- The address changes each time you start it (free quick tunnels). For a fixed address, use a named tunnel with your own domain (Option B).
- Replies are as fast as on your PC: one or two people at a time is fine, many at once will be slow.

## Option B: website on Netlify / Vercel / Cloudflare Pages + server on your PC

Use this if you want a permanent address like `https://my-chat.netlify.app`. The server still runs on your PC, behind a tunnel
with a **fixed** address, because the website has to know where the server is.

1. **Give your PC a fixed public address.** You need a domain in Cloudflare (free account, domain costs money), then create a
   named tunnel, e.g. `https://chat-api.yourdomain.com` → `http://127.0.0.1:8001`
   (Cloudflare dashboard → Zero Trust → Networks → Tunnels, or `cloudflared tunnel create`).
2. **Start the server allowing your website** (PowerShell, in `gpt-oss-chat`):
   ```powershell
   $env:APP_PASSWORD = "your-password"
   $env:ALLOWED_ORIGINS = "https://my-chat.netlify.app"
   .\.venv\Scripts\python.exe server.py
   ```
   (`ALLOWED_ORIGINS` is the exact address of your website, no trailing slash. Separate several with commas.)
3. **Deploy the website** from this GitHub repository and set one environment variable, `API_BASE`, to the server address
   from step 1 (`https://chat-api.yourdomain.com`). Everything else is read from the config files in this repository:

   | Host | What to do |
   |---|---|
   | **Netlify** | Add new site → Import from Git → pick the repo. `netlify.toml` sets the build. Site settings → Environment variables → `API_BASE`. |
   | **Vercel** | Add New → Project → pick the repo. `vercel.json` sets the build. Settings → Environment Variables → `API_BASE`. |
   | **Cloudflare** | Workers & Pages → Create → connect the Git repo. `wrangler.jsonc` tells Cloudflare which folder to publish. In the project's **Settings → Build**, set **Build command** to `node scripts/make-config.js`, keep **Deploy command** as `npx wrangler deploy`, and add `API_BASE` under **Build → Variables and secrets** (build variables, not runtime variables). The `name` in `wrangler.jsonc` must match your Worker's name. If you use classic Cloudflare Pages instead: build command `node scripts/make-config.js`, output directory `gpt-oss-chat/static`. |

   The build fails with a clear message if `API_BASE` is missing or doesn't start with `https://`.
4. Open your site, log in with the password. After changing `API_BASE`, redeploy the site.

## Security (read before sharing a link)

- **Always set a password** when the site is reachable from the internet (`start_public.bat` insists on one). Without `APP_PASSWORD`
  the server is open to everyone who finds the link.
- With a password set, the web-link reader refuses links to your PC and your home network, so visitors can't use it to reach Ollama or your router.
- Keep Ollama itself private: never tunnel port **11434**. Only port **8001** (the chat server) is exposed.
- Visitors use your computer's GPU and disk: generated images are saved in `gpt-oss-chat/data/images` and project files in `gpt-oss-chat/data/projects`.
- Chats are stored in each visitor's own browser; projects and their memory are shared by everyone who logs in with the password.
- The password is shared by everyone. For separate accounts you'd need a real user system, which this app doesn't have.

## Settings reference

| Setting | Meaning |
|---|---|
| `APP_PASSWORD` | Turns on the login. Empty = no login (local use only). |
| `ALLOWED_ORIGINS` | Websites allowed to call the server from a browser (Option B only). |
| `ALLOW_PRIVATE_URLS` | `1` lets the link reader open local addresses. Default `0` with a password, `1` without. |
| `HOST` / `PORT` | Where the server listens. Default `127.0.0.1:8001`. Leave HOST alone when using a tunnel. |
| `API_BASE` | Build setting for Netlify / Vercel / Cloudflare Pages: the server's public `https://` address. |
