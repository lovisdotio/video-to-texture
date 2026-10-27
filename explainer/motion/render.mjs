// Deterministic frame capture: node render.mjs [--from N] [--to N] [--step N] [--out frames]
import puppeteer from "puppeteer-core";
import fs from "node:fs";
import path from "node:path";

const args = Object.fromEntries(process.argv.slice(2).join(" ").split("--").filter(Boolean).map((s) => s.trim().split(/\s+/)));
const root = path.dirname(new URL(import.meta.url).pathname);
const out = path.resolve(root, args.out || "frames");
fs.mkdirSync(out, { recursive: true });
const counts = {};
for (const d of fs.readdirSync(path.join(root, "assets/seq"))) counts[d] = fs.readdirSync(path.join(root, "assets/seq", d)).filter((f) => f.endsWith(".jpg")).length;

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  args: ["--allow-file-access-from-files", "--disable-web-security", "--force-color-profile=srgb", "--hide-scrollbars"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
await page.evaluateOnNewDocument((c) => (window.SEQ_COUNTS = c), counts);
await page.goto("file://" + path.join(root, "index.html"), { waitUntil: "load" });
const total = Math.round(await page.evaluate(() => window.DURATION * window.FPS));
const from = +(args.from || 0), to = +(args.to || total), step = +(args.step || 1);
for (let f = from; f < to; f += step) {
  await page.evaluate((f) => window.seek(f), f);
  await page.screenshot({ path: path.join(out, `${String(f).padStart(5, "0")}.jpg`), type: "jpeg", quality: 95 });
  if (f % 60 === 0) console.log("frame", f, "/", total);
}
await browser.close();
