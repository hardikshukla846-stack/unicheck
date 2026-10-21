// face.js - shared face helper for index.html (sign-up) and library.html (check-in).
// Everything runs in the student's browser. Only a list of 128 numbers (a "face print") is
// saved, never a photo.

const LIBS = [
  { script: "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js",
    models: "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights" },
  { script: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/dist/face-api.js",
    models: "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model" }
];

// lower = stricter. 0.6 is the face-api.js default; 0.5 rejects many genuine matches.
export const FACE_THRESHOLD = 0.55;

let ready = null;
function addScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.onload = res; s.onerror = () => { s.remove(); rej(new Error("script")); };
    document.head.appendChild(s);
  });
}

export function loadFace() {
  if (ready) return ready;
  ready = (async () => {
    for (const L of LIBS) {
      try {
        await addScript(L.script);
        const f = window.faceapi;
        await Promise.all([
          f.nets.tinyFaceDetector.loadFromUri(L.models),
          f.nets.faceLandmark68Net.loadFromUri(L.models),   // full landmark model: better face alignment
          f.nets.faceRecognitionNet.loadFromUri(L.models)
        ]);
        return;
      } catch (e) { console.error("face library source failed:", e); }
    }
    throw new Error("face-lib");
  })();
  ready.catch(() => { ready = null; });
  return ready;
}

export async function openCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  return stream;
}
export function closeCamera(stream) { if (stream) stream.getTracks().forEach(t => t.stop()); }

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const ear = p => (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]));
const sleep = ms => new Promise(r => setTimeout(r, ms));

export async function scanFace(video, status, control) {
  const f = window.faceapi, opts = new f.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
  const t0 = Date.now(); const recent = []; let closed = false, blinked = false;
  status("Look at the camera and blink once");
  while (Date.now() - t0 < 25000) {
    if (control && control.cancelled) throw new Error("cancelled");
    const det = await f.detectSingleFace(video, opts).withFaceLandmarks();
    if (!det) continue;
    const e = (ear(det.landmarks.getLeftEye()) + ear(det.landmarks.getRightEye())) / 2;
    recent.push(e); if (recent.length > 12) recent.shift();   // rolling baseline, not all-time max
    const base = Math.max(...recent);
    if (e < base * 0.75) closed = true;
    else if (closed && e > base * 0.9) { blinked = true; break; }
  }
  if (!blinked) throw new Error("no-blink");

  status("Hold still, look straight at the camera…");
  await sleep(400);                       // let eyes/face settle after the blink
  const out = [], tEnd = Date.now() + 6000;
  while (Date.now() < tEnd && out.length < 4) {
    if (control && control.cancelled) throw new Error("cancelled");
    const r = await f.detectSingleFace(video, opts).withFaceLandmarks().withFaceDescriptor();
    if (r && r.detection.score > 0.7) out.push(Array.from(r.descriptor));
    await sleep(150);                     // space samples out so averaging actually helps
  }
  if (out.length < 3) throw new Error("no-face");
  return out[0].map((_, i) => Math.round((out.reduce((s, d) => s + d[i], 0) / out.length) * 1e4) / 1e4);
}

export const faceDistance = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));

export function faceError(err) {
  switch (err && err.message) {
    case "cancelled": return "";
    case "no-blink": return "Couldn't see you blink. Face the camera in good light and try again.";
    case "no-face": return "Couldn't read your face. Try again in better light.";
    case "face-lib": return "Couldn't load face recognition. Check your connection or try another network.";
    default: return "Couldn't use the camera. Allow camera access and open the site over https.";
  }
}