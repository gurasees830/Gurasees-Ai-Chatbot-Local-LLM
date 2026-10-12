// Build step for Netlify / Vercel / Cloudflare Pages: writes gpt-oss-chat/static/config.js so the page knows where
// your chat server (the PC running Ollama, reached through a tunnel) lives.
//
//   API_BASE=https://chat-api.example.com node scripts/make-config.js
const fs = require("fs");
const path = require("path");

const raw = (process.env.API_BASE || "").trim();
if (!raw) {
  console.error(
    "API_BASE is not set.\n" +
    "Add an environment variable API_BASE in your hosting dashboard, set to the public https:// address of your chat server\n" +
    "(for example https://chat-api.yourdomain.com). See HOSTING.md."
  );
  process.exit(1);
}
let url;
try { url = new URL(raw); } catch { console.error(`API_BASE is not a valid address: ${raw}`); process.exit(1); }
if (url.protocol !== "https:") {
  console.error("API_BASE must start with https:// (browsers block a secure website from calling a plain http:// server).");
  process.exit(1);
}
const apiBase = url.origin;
const out = path.join(__dirname, "..", "gpt-oss-chat", "static", "config.js");
fs.writeFileSync(out, `window.APP_CONFIG = { apiBase: ${JSON.stringify(apiBase)} };\n`);
console.log(`config.js written: apiBase = ${apiBase}`);
