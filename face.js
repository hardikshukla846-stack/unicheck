// face.js - shared face helper for index.html (sign-up), library.html (check-in) and facetest.html.
// Everything runs in the student's browser. Only a list of 128 numbers (a "face print") is
// saved, never a photo. Functions that take a callback call it as (text, percent).

const ORIGINAL = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js";
// Sources are tried in order until one works completely (code + models).
const LIBS = [
  { script: ORIGINAL, models: "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights" },
  { script: ORIGINAL, models: "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights" },
  { script: "https://unpkg.com/face-api.js@0.22.2/dist/face-api.min.js", models: "https://unpkg.com/face-api.js@0.22.2/weights" },
  { script: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/dist/face-api.js", models: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model" }
];

export const FACE_THRESHOLD = 0.5;   // lower = stricter. Distances above this are "not the same person".
const SCAN_SECONDS = 20;             // how long to wait for a blink before giving up
const SAMPLES = 2;                   // face prints averaged (more = steadier but slower)

let ready = null, loadedSrc = null;
export let sourceUsed = "";
const reporters = new Set();
let last = null;
function emit(text, pct) { last = [text, pct]; reporters.forEach(r => { try { r(text, pct); } catch {} }); }

function addScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.onload = res; s.onerror = () => { s.remove(); rej(new Error("could not download " + src)); };
    document.head.appendChild(s);
  });
}

// Loads the recognition code and models once. Safe to call early (for example in the
// background) and again later, because the work is shared.
export function loadFace(report) {
  if (report) { reporters.add(report); if (last) report(last[0], last[1]); }
  if (!ready) {
    ready = (async () => {
      const failures = [];
      for (const L of LIBS) {
        try {
          emit("Downloading face recognition (first time only)…", 12);
          if (loadedSrc !== L.script) { await addScript(L.script); loadedSrc = L.script; }
          const f = window.faceapi;
          if (!f) throw new Error("face-api not found after download");
          emit("Loading face models…", 30);
          let done = 0;
          const step = () => { done++; emit("Loading face models… " + done + "/3", 30 + done * 10); };
          await Promise.all([
            f.nets.tinyFaceDetector.loadFromUri(L.models).then(step),
            f.nets.faceLandmark68TinyNet.loadFromUri(L.models).then(step),
            f.nets.faceRecognitionNet.loadFromUri(L.models).then(step)
          ]);
          emit("Getting ready…", 62);
          try {   // warm-up pass so the first real scan is fast
            const c = document.createElement("canvas"); c.width = c.height = 64;
            await f.detectSingleFace(c, new f.TinyFaceDetectorOptions({ inputSize: 160 })).withFaceLandmarks(true);
          } catch {}
          sourceUsed = L.models;
          return;
        } catch (e) { console.error("face source failed:", L.models, e); failures.push(L.models.replace("https://", "") + " -> " + (e && e.message || e)); }
      }
      const err = new Error("face-lib"); err.detail = failures.join(" | "); throw err;
    })();
    ready.catch(() => { ready = null; });
  }
  return ready.finally(() => { if (report) reporters.delete(report); });
}

export async function openCamera(video) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("no-camera-api");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 480 }, height: { ideal: 360 } }, audio: false
  });
  video.srcObject = stream;
  try { await video.play(); } catch (e) { closeCamera(stream); throw e; }
  for (let i = 0; i < 80 && !video.videoWidth; i++) await new Promise(r => setTimeout(r, 50));
  if (!video.videoWidth) { closeCamera(stream); throw new Error("no-video"); }
  return stream;
}
export function closeCamera(stream) { if (stream) stream.getTracks().forEach(t => t.stop()); }

// Eye openness from 6 landmark points
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const ear = p => (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]));

// Waits for one blink (proves a live person, not a still photo), then returns an averaged face print.
// control = { cancelled: false } lets the caller stop the scan.
export async function scanFace(video, status, control) {
  const f = window.faceapi, opts = new f.TinyFaceDetectorOptions({ inputSize: 160, scoreThreshold: 0.4 });
  const t0 = Date.now();
  let maxEar = 0, closed = false, blinked = false, seen = false;
  status("Look at the camera", 64);
  while (Date.now() - t0 < SCAN_SECONDS * 1000) {
    if (control && control.cancelled) throw new Error("cancelled");
    const left = Math.max(0, SCAN_SECONDS - Math.floor((Date.now() - t0) / 1000));
    const det = await f.detectSingleFace(video, opts).withFaceLandmarks(true);
    if (!det) { status("No face found. Move closer and face the light (" + left + "s)", seen ? 70 : 64); continue; }
    seen = true;
    status("Face found ✓ Now blink once (" + left + "s)", 72);
    const e = (ear(det.landmarks.getLeftEye()) + ear(det.landmarks.getRightEye())) / 2;
    maxEar = Math.max(maxEar * 0.99, e);
    if (e < maxEar * 0.8) closed = true;
    else if (closed && e > maxEar * 0.88) { blinked = true; break; }
  }
  if (!blinked) throw new Error(seen ? "no-blink" : "no-face");
  status("Blink seen ✓ Hold still…", 82);
  const out = [];
  for (let i = 0; i < 25 && out.length < SAMPLES; i++) {
    if (control && control.cancelled) throw new Error("cancelled");
    const r = await f.detectSingleFace(video, opts).withFaceLandmarks(true).withFaceDescriptor();
    if (r) { out.push(Array.from(r.descriptor)); status("Reading your face… " + out.length + "/" + SAMPLES, 82 + out.length * 8); }
  }
  if (out.length < SAMPLES) throw new Error("no-face");
  status("Done ✓", 100);
  return out[0].map((_, i) => Math.round((out.reduce((s, d) => s + d[i], 0) / out.length) * 1e4) / 1e4);
}

export const faceDistance = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));

// Turns an error into a message a student can act on (with the technical reason at the end)
export function faceError(err) {
  const m = err && err.message, n = err && err.name;
  if (m === "cancelled") return "";
  if (m === "no-blink") return "Couldn't see you blink. Face the camera in good light and try again.";
  if (m === "no-face") return "Couldn't see your face clearly. Try again in better light.";
  if (m === "face-lib") return "Couldn't load face recognition. Check your connection or try another network. [" + (err.detail || "") + "]";
  if (m === "no-camera-api") return "The camera isn't available on this address. Open the site over https, or use Live Server on this computer.";
  if (m === "no-video") return "The camera opened but gave no picture. Close other apps using the camera and try again.";
  if (n === "NotAllowedError") return "Camera permission was blocked. Allow the camera in your browser settings and try again.";
  if (n === "NotFoundError") return "No camera was found on this device.";
  if (n === "NotReadableError") return "The camera is being used by another app. Close it and try again.";
  return "Face recognition could not run. [" + (n || "") + " " + (m || err) + "]";
}