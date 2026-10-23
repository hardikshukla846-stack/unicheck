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

export const FACE_JS_VERSION = "fast-3";
export const FACE_THRESHOLD = 0.6;   // lower = stricter. Distances above this are "not the same person".
const SCAN_SECONDS = 20;             // how long to wait for a blink before giving up

const MODEL_FILES = ["tiny_face_detector_model-weights_manifest.json", "tiny_face_detector_model-shard1",
  "face_landmark_68_tiny_model-weights_manifest.json", "face_landmark_68_tiny_model-shard1",
  "face_recognition_model-weights_manifest.json", "face_recognition_model-shard1", "face_recognition_model-shard2"];
const prefetched = new Set();
function prefetchModels(base) {
  if (prefetched.has(base)) return; prefetched.add(base);
  MODEL_FILES.forEach(n => fetch(base + "/" + n).catch(() => {}));
}

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
          prefetchModels(L.models);   // fetch model files while the code downloads, not after
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
          try {   // warm-up: run every network once so the first real scan does not compile GPU code
            const cv = n => { const c = document.createElement("canvas"); c.width = c.height = n; return c; };
            await f.detectSingleFace(cv(160), new f.TinyFaceDetectorOptions({ inputSize: 160 }));
            await f.nets.faceLandmark68TinyNet.detectLandmarks(cv(112));
            await f.nets.faceRecognitionNet.computeFaceDescriptor(cv(150));
          } catch (e) { console.warn("warm-up skipped", e); }
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
// ref (optional) = the stored face print. If it is given, the scan ends as soon as the face clearly matches it.
const mean = arr => arr[0].map((_, i) => arr.reduce((s, d) => s + d[i], 0) / arr.length);
const round4 = d => d.map(v => Math.round(v * 1e4) / 1e4);
const sleep = ms => new Promise(r => setTimeout(r, ms));
export async function scanFace(video, status, control, ref) {
  const f = window.faceapi, opts = new f.TinyFaceDetectorOptions({ inputSize: 160, scoreThreshold: 0.4 });
  const t0 = Date.now(), recent = [];
  let closed = false, blinked = false, seen = false;
  status("Look at the camera", 64);
  while (Date.now() - t0 < SCAN_SECONDS * 1000) {
    if (control && control.cancelled) throw new Error("cancelled");
    const left = Math.max(0, SCAN_SECONDS - Math.floor((Date.now() - t0) / 1000));
    const det = await f.detectSingleFace(video, opts).withFaceLandmarks(true);
    if (!det) { status("No face found. Move closer and face the light (" + left + "s)", seen ? 70 : 64); continue; }
    seen = true;
    status("Face found ✓ Now blink once (" + left + "s)", 72);
    const e = (ear(det.landmarks.getLeftEye()) + ear(det.landmarks.getRightEye())) / 2;
    recent.push(e); if (recent.length > 15) recent.shift();      // baseline = widest eyes in the last ~15 frames
    const base = Math.max(...recent);
    if (e < base * 0.85) closed = true;
    else if (closed && e > base * 0.92) { blinked = true; break; }
  }
  if (!blinked) throw new Error(seen ? "no-blink" : "no-face");
  status("Blink seen ✓ Hold still…", 82);
  await sleep(100);                                              // let the eyes finish opening
  const out = [], tEnd = Date.now() + 6000;
  while (Date.now() < tEnd && out.length < 6) {
    if (control && control.cancelled) throw new Error("cancelled");
    const r = await f.detectSingleFace(video, opts).withFaceLandmarks(true).withFaceDescriptor();
    if (!r) continue;
    const d = Array.from(r.descriptor); out.push(d);
    status("Reading your face… " + out.length, Math.min(98, 82 + out.length * 4));
    if (ref) {   // verification: stop early on a clear match, otherwise keep sampling and average
      if (out.length === 1 && faceDistance(d, ref) <= FACE_THRESHOLD - 0.1) { status("Done ✓", 100); return round4(d); }
      if (out.length >= 2) {   // any two consecutive prints that average close to the saved face = same person
        const pair = mean(out.slice(-2));
        if (faceDistance(pair, ref) <= FACE_THRESHOLD - 0.03) { status("Done ✓", 100); return round4(pair); }
      }
    }
  }
  if (out.length < (ref ? 2 : 3)) throw new Error("no-face");
  let use = out;
  if (!ref && out.length >= 4) {   // registration: drop the one blurry/odd sample so the saved print is clean
    const m = mean(out); let worst = 0;
    out.forEach((d, i) => { if (faceDistance(d, m) > faceDistance(out[worst], m)) worst = i; });
    use = out.filter((_, i) => i !== worst);
  }
  status("Done ✓", 100);
  return round4(mean(use));
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