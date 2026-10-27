// Explainer film timeline. Everything is a pure function of time so render.mjs can seek frame by frame.
const FPS = 30;
const BEAT = 0.5; // 120 BPM grid
const SEQ = window.SEQ_COUNTS || {}; // filled by render.mjs / preview server: { name: frameCount }
const $ = (s) => document.querySelector(s);

// scene windows [start, end) in seconds
const SCENES = {
  s0: [0, 2], s1: [2, 6], s2: [6, 10], s3: [10, 16], s4: [16, 21], s5: [21, 24], s5z: [24, 26.5], s6: [26.5, 31], s7: [31, 35], s8: [35, 42],
};
const DURATION = 42;

// which film time each sequence element starts playing at (by scene); reverse/hold handled per element
for (const [id, [a]] of Object.entries(SCENES)) {
  document.querySelectorAll(`#${id} img.seq`).forEach((img) => (img.dataset.start = a));
}

const tl = gsap.timeline({ paused: true, defaults: { ease: "expo.out" } });
gsap.set("#chip", { opacity: 0 });
const flash = (t, o = 0.85) => tl.fromTo("#flash", { opacity: o }, { opacity: 0, duration: 0.25, ease: "power2.out", immediateRender: false }, t);

// steps: chip text + HUD counter are pure functions of time (set in seek)
const STEPS = [];
function chip(t, n, model, what, note, out) {
  STEPS.push({ t, out, n, model, what, note: note || "" });
  tl.fromTo("#chip", { x: -80, opacity: 0, clipPath: "inset(0 100% 0 0)" }, { x: 0, opacity: 1, clipPath: "inset(0 0% 0 0)", duration: 0.45, immediateRender: false }, t)
    .to("#chip", { opacity: 0, x: -30, duration: 0.2, ease: "power2.in" }, out - 0.2);
}
const HUD = [];
function hud(t, n) { HUD.push([t, n]); }
function applySteps(t) {
  const s = STEPS.filter((x) => t >= x.t).pop();
  if (s) { $("#chipN").textContent = s.n; $("#chipM").textContent = s.model; $("#chipW").textContent = s.what; $("#chipX").textContent = s.note; }
  const h = HUD.filter(([a]) => t >= a).pop(); const n = h ? h[1] : 0;
  $("#hudR").textContent = `${String(n).padStart(2, "0")} / 06`;
  document.querySelectorAll("#bar i").forEach((b, i) => b.style.setProperty("--p", i < n ? 1 : 0));
}

// S0 — cold open, word per beat
tl.fromTo("#s0k", { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5 }, 0.1)
  .fromTo("#s0a", { opacity: 0, y: 120, skewY: 6 }, { opacity: 1, y: 0, skewY: 0, duration: 0.6 }, 0.25)
  .fromTo("#s0b", { opacity: 0, y: 120, skewY: 6 }, { opacity: 1, y: 0, skewY: 0, duration: 0.6 }, 0.25 + BEAT * 1.5)
  .to(["#s0a", "#s0b", "#s0k"], { y: -60, opacity: 0, duration: 0.3, ease: "power3.in", stagger: 0.04 }, 1.7);
tl.set(["#hud", "#bar"], { opacity: 0 }, 0).to(["#hud", "#bar"], { opacity: 1, duration: 0.4 }, 2);

// S1 — two worlds
flash(2, 0.6);
tl.fromTo("#s1l", { clipPath: "inset(0 0 100% 0)" }, { clipPath: "inset(0 0 0% 0)", duration: 0.6 }, 2)
  .fromTo("#s1r", { clipPath: "inset(100% 0 0 0)" }, { clipPath: "inset(0% 0 0 0)", duration: 0.6 }, 2 + BEAT)
  .fromTo("#s1la", { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.4 }, 2.3)
  .fromTo("#s1ra", { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.4 }, 2.8)
  .fromTo(["#s1l img", "#s1r img"], { filter: "brightness(1)" }, { filter: "brightness(0.42)", immediateRender: false, duration: 0.5, ease: "power2.out" }, 4)
  .fromTo("#s1q", { opacity: 0, scale: 1.12, filter: "blur(16px)" }, { opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.7 }, 4)
  .to(["#s1l", "#s1r"], { scale: 0.96, duration: 2, ease: "none" }, 4);

// S2 — 01 Blender orbit / 02 Seedance on the same camera
flash(6);
hud(6, 1);
chip(6, "01", "Blender", "Clay scene · orbit with known camera poses", "", 8);
tl.fromTo("#s2wipe", { clipPath: "inset(0 100% 0 0)" }, { clipPath: "inset(0 0% 0 0)", duration: 1.6, ease: "power2.inOut" }, 7.5)
  .fromTo("#s2scan", { left: 0, opacity: 1 }, { left: 1916, duration: 1.6, ease: "power2.inOut" }, 7.5)
  .set("#s2scan", { opacity: 0 }, 9.1);
hud(8, 2);
chip(8, "02", "Seedance 2.5", "Reference-to-video · same motion, new look", "", 10);

// S3 — 03 nvdiffrast fit
flash(10);
hud(10, 3);
chip(10, "03", "nvdiffrast", "Differentiable texture fit · every frame at once", "Deployable on fal Serverless · H100 · contact the team", 16);
tl.fromTo("#s3uv", { opacity: 0, scale: 0.9, rotate: -3 }, { opacity: 1, scale: 1, rotate: 0, duration: 0.6 }, 12.5)
  .fromTo("#s3l0", { opacity: 0 }, { opacity: 1, duration: 0.3 }, 12.8)
  .fromTo("#s3l1", { opacity: 0 }, { opacity: 1, duration: 0.3 }, 11)
  .fromTo("#s3uv1", { opacity: 0 }, { opacity: 1, duration: 2.0, ease: "power1.inOut" }, 13)
  .set("#s3it", { textContent: "fitting…" }, 13)
  .set("#s3it", { textContent: "fitted" }, 14.5)
  .fromTo("#s3uv", { scale: 1 }, { scale: 1.06, duration: 1.5, ease: "power1.inOut" }, 14.5);

// S4 — 04 re-unwrap (unfold), then S5 — 05 Seedream fills the atlas
flash(16);
hud(16, 4);
chip(16, "04", "Re-unwrap", "Clean UV islands · the body unfolds to its atlas", "", 21);
flash(16.6, 0.35);
hud(21, 5);
chip(21, "05", "Seedream 5 Pro", "Completes seams & gaps · Topaz upscale to 8K", "", 26.5);
flash(21);
tl.fromTo("#s5zoom", { scale: 1.15 }, { scale: 1.7, duration: 3, ease: "power1.inOut", immediateRender: false }, 21)
  .fromTo(["#s5a", "#s5b"], { opacity: 0 }, { opacity: 1, duration: 0.3, immediateRender: false }, 21.2)
  .fromTo("#s5wipe", { clipPath: "inset(0 100% 0 0)" }, { clipPath: "inset(0 0% 0 0)", duration: 1.8, ease: "power2.inOut" }, 21.5)
  .fromTo("#s5scan", { left: 0, opacity: 1 }, { left: 1916, duration: 1.8, ease: "power2.inOut", immediateRender: false }, 21.5)
  .set("#s5scan", { opacity: 0 }, 23.3);
// loupes: same camera, same frame — video-only texture vs final
flash(24, 0.5);
tl.fromTo("#lpA", { scale: 0, rotate: -20 }, { scale: 1, rotate: 0, duration: 0.6, ease: "back.out(1.6)" }, 24)
  .fromTo("#lpB", { scale: 0, rotate: 20 }, { scale: 1, rotate: 0, duration: 0.6, ease: "back.out(1.6)" }, 24 + BEAT)
  .set(["#lpAl", "#lpBl"], { xPercent: -50 }, 0)
  .fromTo("#lpH", { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.4 }, 24.1)
  .fromTo(["#lpAl", "#lpBl"], { opacity: 0 }, { opacity: 1, duration: 0.3, stagger: BEAT }, 24.3)
  .fromTo([".loupe img"], { scale: 2.0 }, { scale: 2.6, duration: 2.5, ease: "none" }, 24);

// S6 — 06 fold back onto the mesh
flash(26.5);
hud(26.5, 6);
chip(26.5, "06", "Back on the mesh", "Real UVs · real-time 3D · any renderer", "", 31);

// S7 — three-way compare
flash(31);
tl.fromTo("#s7c", { clipPath: "inset(0 100% 0 0)" }, { clipPath: "inset(0 66.66% 0 0)", duration: 0.5 }, 31)
  .fromTo("#s7v", { clipPath: "inset(0 66.66% 0 33.33%)", opacity: 0 }, { opacity: 1, duration: 0.01 }, 31)
  .fromTo("#s7v", { clipPath: "inset(0 100% 0 33.33%)" }, { clipPath: "inset(0 33.33% 0 33.33%)", duration: 0.5 }, 31 + BEAT)
  .fromTo("#s7f", { clipPath: "inset(0 0 0 100%)" }, { clipPath: "inset(0 0 0 66.66%)", duration: 0.5 }, 31 + BEAT * 2)
  .fromTo(["#s7lc", "#s7lv", "#s7lf"], { opacity: 0, y: 12 }, { opacity: 1, y: 0, duration: 0.3, stagger: BEAT }, 31.1)
  .to("#s7c", { clipPath: "inset(0 100% 0 0)", duration: 0.6, ease: "power3.inOut" }, 33.5)
  .to("#s7v", { clipPath: "inset(0 100% 0 0%)", duration: 0.6, ease: "power3.inOut" }, 33.5)
  .to("#s7f", { clipPath: "inset(0 0 0 0%)", duration: 0.6, ease: "power3.inOut" }, 33.5)
  .to(["#s7lc", "#s7lv", "#s7lf"], { opacity: 0, duration: 0.2 }, 33.4);

// S8 — hero + recap
tl.to(["#hud", "#bar"], { opacity: 0, duration: 0.4 }, 35);
tl.fromTo("#pipe .s, #pipe .ar", { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.4, stagger: BEAT / 2 }, 37)
  .fromTo("#endt", { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.6 }, 39)
  .to("#s8 img.seq", { filter: "brightness(0.55)", duration: 1.5, ease: "power1.inOut" }, 37);

tl.set({}, {}, DURATION);

// --- sequence handling ---------------------------------------------------
const pad = (n) => String(n).padStart(4, "0");
function seqFrame(img, t) {
  const name = img.dataset.seq, count = SEQ[name] || 1;
  let i = Math.floor((t - +img.dataset.start) * FPS + 1e-6);
  if (img.dataset.hold) i = +img.dataset.hold;
  i = Math.max(0, Math.min(count - 1, i));
  if (img.dataset.reverse) i = count - 1 - i;
  return `assets/seq/${name}/${pad(i)}.jpg`;
}

window.seek = async function (frame) {
  const t = frame / FPS;
  for (const [id, [a, b]] of Object.entries(SCENES)) $("#" + id).style.visibility = t >= a && t < b ? "visible" : "hidden";
  tl.seek(t, false);
  applySteps(t);
  const pending = [];
  document.querySelectorAll("img.seq").forEach((img) => {
    if (getComputedStyle(img).visibility === "hidden") return;
    const src = seqFrame(img, t);
    if (img.getAttribute("src") !== src) { img.setAttribute("src", src); pending.push(img.decode().catch(() => {})); }
  });
  await Promise.all(pending);
  await document.fonts.ready;
};
window.DURATION = DURATION;
window.FPS = FPS;
