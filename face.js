// face.js - shared face helper for index.html (sign-up) and library.html (check-in).
// Everything runs in the student's browser. Only a list of 128 numbers (a "face print") is
// saved, never a photo.
//
// Every function that takes a `report` / `status` callback calls it as (text, percent).

// Two sources are tried in turn, each with its own matching model files.
const LIBS = [
  { script: "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js",
    models: "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights" },
  { script: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/dist/face-api.js",
    models: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model" }
];

export const FACE_THRESHOLD = 0.5;   // lower = stricter. Distances above this are "not the same person".
const SCAN_SECONDS = 20;             // how long to wait for a blink before giving up
const SAMPLES = 2;                   // face prints averaged (more = steadier but slower)

let ready = null;
const reporters = new Set();
let last = null;
function emit(text, pct) { last = [text, pct]; reporters.forEach(r => { try { r(text, pct); } catch {} }); }

function addScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.onload = res; s.onerror = () => { s.remove(); rej(new Error("script")); };
    document.head.appendChild(s);
  });
}

// Loads the recognition code and models once. Safe to call early (for example in the
// background) and again later, because the work is shared.
export function loadFace(report) {
  if (report) { reporters.add(report); if (last) report(last[0], last[1]); }
  if (ready) return ready.finally(() => { if (report) reporters.delete(report); });
  ready = (async () => {
    for (const L of LIBS) {
      try {
        emit("Downloading face recognition (first time only)…", 12);
        await addScript(L.script);
        const f = window.faceapi;
        emit("Loading face models…", 30);
        let done = 0;
        const step = () => { done++; emit("Loading face models… " + done + "/3", 30 + done * 10); };
        await Promise.all([
          f.nets.tinyFaceDetector.loadFromUri(L.models).then(step),
          f.nets.faceLandmark68TinyNet.loadFromUri(L.models).then(step),
          f.nets.faceRecognitionNet.loadFromUri(L.models).then(step)
        ]);
        emit("Getting ready…", 62);
        // Warm-up pass so the first real scan is fast
        try {
          const c = document.createElement("canvas"); c.width = c.height = 64;
          await f.detectSingleFace(c, new f.TinyFaceDetectorOptions({ inputSize: 160 })).withFaceLandmarks(true);
        } catch {}
        return;
      } catch (e) { console.error("face library source failed:", e); }
    }
    throw new Error("face-lib");
  })();
  ready.catch(() => { ready = null; });
  return ready.finally(() => { if (report) reporters.delete(report); });
}

export async function openCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 480 }, height: { ideal: 360 } }, audio: false
  });
  video.srcObject = stream;
  await video.play();
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

export function faceError(err) {
  switch (err && err.message) {
    case "cancelled": return "";
    case "no-blink": return "Couldn't see you blink. Face the camera in good light and try again.";
    case "no-face": return "Couldn't see your face clearly. Try again in better light.";
    case "face-lib": return "Couldn't load face recognition. Check your connection or try another network.";
    default: return "Couldn't use the camera. Allow camera access and open the site over https.";
  }
}