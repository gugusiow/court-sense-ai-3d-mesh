import { Component, useEffect, useMemo, useRef, useState, Suspense } from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Box3, Vector3 } from "three";
import { GLTFLoader, OBJLoader, PLYLoader } from "three-stdlib";
import "./App.css";

// The dev server on this computer forwards /api to the backend.
const API = "";
const TOKEN_KEY = "courtsense-token";
// Drop the how-to-record clip at frontend/public/recording-guide.mp4
const GUIDE_VIDEO = "/recording-guide.mp4";
// Drop the one mesh at frontend/public/demo/mesh.ply (.obj or .glb also work if you change this name)
const DEMO_MESH = "/demo/mesh.ply";

const FEATURE_LABELS = {
  mean_elbow_angle: "Mean elbow angle",
  wrist_height_range: "Wrist height range",
};

// World height of a fitted mesh. The canvas is the same 9:16 frame as the video,
// and the camera is placed so this height fills that frame the way a full body fills the clip.
const MESH_FRAME_HEIGHT = 1.7;
const MESH_CAMERA = { position: [0, 0, 3.2], fov: 40 };

const STATUS_ORDER = ["received", "analyzing", "ready"];
const STATUS_LABELS = {
  received: "Video received",
  analyzing: "Running analysis",
  ready: "Comparison ready",
};

function formatFeature(feature) {
  if (!feature.valid || feature.value == null) return "Unavailable";
  const rounded = Number(feature.value).toFixed(feature.units === "degrees" ? 1 : 2);
  if (feature.units === "degrees") return `${rounded}°`;
  if (feature.units === "metres") return `${rounded} m`;
  return `${rounded} ${feature.units}`;
}

async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg).filter(Boolean).join(", ");
    }
  } catch {
    // The body was not JSON.
  }
  return `API error ${response.status}`;
}

async function api(path, token, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(`${API}${path}`, { ...options, headers });
  if (!response.ok) throw new Error(await readError(response));
  if (response.status === 204) return null;
  return response.json();
}

function BasketballLogo() {
  return (
    <svg className="basketball" viewBox="0 0 64 64" aria-hidden="true">
      <circle cx="32" cy="32" r="30" fill="#f97316" />
      <path
        d="M32 2.5c7.5 9 7.5 50 0 59M32 2.5c-7.5 9-7.5 50 0 59M3 32h58M9.5 16.5c13 7 32 7 45 0M9.5 47.5c13-7 32-7 45 0"
        fill="none"
        stroke="#7c2d12"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

function Brand() {
  return (
    <span className="brand">
      <BasketballLogo />
      <strong>CourtSense</strong>
    </span>
  );
}

class MeshBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return <p>Put your mesh at frontend/public/demo/mesh.ply, or change DEMO_MESH in frontend/src/App.js.</p>;
    }
    return this.props.children;
  }
}

function PlyBody({ url }) {
  const source = useLoader(PLYLoader, url);
  const fitted = useMemo(() => {
    const geometry = source.clone();
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const height = Math.max(box.max.y - box.min.y, 0.001);
    geometry.translate(
      -((box.min.x + box.max.x) / 2),
      -((box.min.y + box.max.y) / 2),
      -((box.min.z + box.max.z) / 2),
    );
    return { geometry, scale: MESH_FRAME_HEIGHT / height };
  }, [source]);
  return (
    <mesh geometry={fitted.geometry} scale={fitted.scale}>
      <meshStandardMaterial color="#f4f1de" roughness={0.55} metalness={0.05} />
    </mesh>
  );
}

function fittedObject(source) {
  const clone = source.clone(true);
  clone.position.set(0, 0, 0);
  clone.rotation.set(0, 0, 0);
  clone.scale.set(1, 1, 1);
  clone.updateMatrixWorld(true);
  const box = new Box3().setFromObject(clone);
  const size = box.getSize(new Vector3());
  const center = box.getCenter(new Vector3());
  clone.position.sub(center);
  const height = Math.max(size.y, 0.001);
  clone.scale.setScalar(MESH_FRAME_HEIGHT / height);
  return clone;
}

function ObjBody({ url }) {
  const source = useLoader(OBJLoader, url);
  const fitted = useMemo(() => fittedObject(source), [source]);
  return <primitive object={fitted} />;
}

function GltfBody({ url }) {
  const loaded = useLoader(GLTFLoader, url);
  const fitted = useMemo(() => fittedObject(loaded.scene), [loaded]);
  return <primitive object={fitted} />;
}

function DemoMesh() {
  const ext = DEMO_MESH.split(".").pop().toLowerCase();
  if (ext === "obj") return <ObjBody url={DEMO_MESH} />;
  if (ext === "glb" || ext === "gltf") return <GltfBody url={DEMO_MESH} />;
  return <PlyBody url={DEMO_MESH} />;
}

function MeshPane() {
  return (
    <div className="mesh-pane">
      <h2>Mesh</h2>
      <div className="media-frame">
        <MeshBoundary>
          <Canvas camera={MESH_CAMERA}>
            <color attach="background" args={["#111111"]} />
            <ambientLight intensity={0.7} />
            <directionalLight position={[2, 3, 4]} intensity={1.2} />
            <Suspense fallback={null}>
              <DemoMesh />
            </Suspense>
            <OrbitControls enablePan={false} enableZoom={false} />
          </Canvas>
        </MeshBoundary>
      </div>
      <p className="mesh-note">Same frame as the video. Drag to turn it.</p>
    </div>
  );
}

function Progress({ status }) {
  const active = STATUS_ORDER.indexOf(status);
  return (
    <ol className="progress">
      {STATUS_ORDER.map((step, index) => (
        <li key={step} className={index <= active ? "done" : ""}>
          {STATUS_LABELS[step]}
        </li>
      ))}
    </ol>
  );
}

function AuthScreen({ onAuth }) {
  const [mode, setMode] = useState("login");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    const form = new FormData(event.currentTarget);
    const body = Object.fromEntries(form.entries());
    if (mode === "signup") {
      body.age = Number(body.age);
      body.height_cm = Number(body.height_cm);
      body.weight_kg = Number(body.weight_kg);
    }
    try {
      const result = await api(mode === "login" ? "/api/login" : "/api/signup", null, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      onAuth(result.access_token, result.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="setup-screen">
      <form className="card" onSubmit={submit}>
        <span className="brand brand-mark"><BasketballLogo /></span>
        <p className="eyebrow">{mode === "login" ? "Welcome back" : "Create an account"}</p>
        <h1>CourtSense</h1>
        {mode === "signup" ? (
          <>
            <label className="field">First name<input name="first_name" required /></label>
            <label className="field">Last name<input name="last_name" required /></label>
          </>
        ) : null}
        <label className="field">Email<input name="email" type="email" required /></label>
        <label className="field">
          Password
          <input name="password" type="password" minLength={mode === "signup" ? 8 : undefined} required />
        </label>
        {mode === "signup" ? (
          <>
            <label className="field">Age<input name="age" type="number" min="1" max="120" required /></label>
            <label className="field">Height (cm)<input name="height_cm" type="number" min="1" step="0.1" required /></label>
            <label className="field">Weight (kg)<input name="weight_kg" type="number" min="1" step="0.1" required /></label>
          </>
        ) : null}
        {error ? <p className="alert" role="alert">{error}</p> : null}
        <button type="submit" disabled={submitting}>{mode === "login" ? "Log in" : "Sign up"}</button>
        <button
          type="button"
          className="secondary auth-switch"
          onClick={() => {
            setMode(mode === "login" ? "signup" : "login");
            setError(null);
          }}
        >
          {mode === "login" ? "Need an account? Sign up" : "Have an account? Log in"}
        </button>
      </form>
    </main>
  );
}

const GUIDE_STEPS = [
  "Place the camera to your side, about chest height.",
  "Keep your whole body in frame, from feet to hands.",
  "Start still. Record one jump shot.",
  "After the ball leaves your hand, keep your shooting arm up.",
  "Use enough light that your arms are visible against the background.",
];

function GuideTab() {
  return (
    <div className="page">
      <section className="panel">
        <p className="eyebrow">Video guide</p>
        <h1>How to record</h1>
        <p>Hold the phone upright and record vertical, from the side. A sideways video cuts off the body.</p>
        <ol className="steps">
          {GUIDE_STEPS.map((text, index) => (
            <li key={text}>
              <span className="num" aria-hidden="true">{index + 1}</span>
              <span>{text}</span>
            </li>
          ))}
        </ol>
        <div className="clip-frame">
          <video className="guide-video portrait" controls src={GUIDE_VIDEO} />
        </div>
      </section>
    </div>
  );
}

function videoFile(blob, mimeType) {
  const type = mimeType || blob.type || "video/webm";
  const extension = type.includes("mp4") ? "mp4" : "webm";
  return new File([blob], `jump-shot.${extension}`, { type });
}

function waitFor(target, eventName) {
  return new Promise((resolve, reject) => {
    const finish = () => {
      target.removeEventListener(eventName, finish);
      target.removeEventListener("error", fail);
      resolve();
    };
    const fail = () => {
      target.removeEventListener(eventName, finish);
      target.removeEventListener("error", fail);
      reject(new Error("This video could not be read."));
    };
    target.addEventListener(eventName, finish);
    target.addEventListener("error", fail);
  });
}

async function trimFile(file, start, end) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.hidden = true;
  video.src = url;
  document.body.appendChild(video);
  try {
    if (video.readyState < 1) await waitFor(video, "loadedmetadata");
    if (typeof video.captureStream !== "function") {
      throw new Error("This browser cannot trim a video. You can still analyse the full clip.");
    }
    if (Math.abs(video.currentTime - start) > 0.05) {
      video.currentTime = start;
      await waitFor(video, "seeked");
    }
    const stream = video.captureStream();
    const mimeType = MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "";
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    const stopped = waitFor(recorder, "stop");
    recorder.start();
    await video.play();
    await new Promise((resolve) => {
      const timer = setInterval(() => {
        if (video.currentTime >= end || video.ended) {
          clearInterval(timer);
          video.pause();
          resolve();
        }
      }, 100);
    });
    recorder.stop();
    await stopped;
    stream.getTracks().forEach((track) => track.stop());
    return videoFile(new Blob(chunks, { type: recorder.mimeType || "video/webm" }), recorder.mimeType);
  } finally {
    video.remove();
    URL.revokeObjectURL(url);
  }
}

function UploadTab({ token, onStarted }) {
  const previewRef = useRef(null);
  const fileRef = useRef(null);
  const recorderRef = useRef(null);
  const [clip, setClip] = useState(null);
  const [clipUrl, setClipUrl] = useState(null);
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [startAt, setStartAt] = useState(0);
  const [endAt, setEndAt] = useState(0);
  const [trimming, setTrimming] = useState(false);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!clip) {
      setClipUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(clip);
    setClipUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [clip]);

  useEffect(() => () => {
    const recorder = recorderRef.current;
    if (recorder) recorder.onstop = null;
    previewRef.current?.srcObject?.getTracks().forEach((track) => track.stop());
    if (recorder?.state === "recording") recorder.stop();
  }, []);

  function clearFileInput() {
    if (fileRef.current) fileRef.current.value = "";
  }

  function deleteClip() {
    const recorder = recorderRef.current;
    if (recorder?.state === "recording") {
      recorder.onstop = null;
      recorder.stop();
    }
    previewRef.current?.srcObject?.getTracks().forEach((track) => track.stop());
    if (previewRef.current) previewRef.current.srcObject = null;
    setRecording(false);
    setClip(null);
    setDuration(0);
    setStartAt(0);
    setEndAt(0);
    clearFileInput();
  }

  async function startRecording() {
    setError(null);
    setClip(null);
    setDuration(0);
    clearFileInput();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      setRecording(true);
      previewRef.current.srcObject = stream;
      await previewRef.current.play();
      const mimeType = MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "";
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      const chunks = [];
      const startedAt = performance.now();
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        if (previewRef.current) previewRef.current.srcObject = null;
        const seconds = Math.max((performance.now() - startedAt) / 1000, 0.1);
        setDuration(seconds);
        setStartAt(0);
        setEndAt(seconds);
        setClip(videoFile(new Blob(chunks, { type: recorder.mimeType || "video/webm" }), recorder.mimeType));
        setRecording(false);
      };
      recorder.start();
      recorderRef.current = recorder;
    } catch {
      previewRef.current?.srcObject?.getTracks().forEach((track) => track.stop());
      setRecording(false);
      setError("Allow the camera, then record again.");
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }

  function chooseFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(null);
    setDuration(0);
    setStartAt(0);
    setEndAt(0);
    setClip(file);
  }

  function noteLength(event) {
    const length = event.currentTarget.duration;
    if (!Number.isFinite(length) || length <= 0) return;
    setDuration(length);
    setStartAt((current) => Math.min(current, Math.max(length - 0.1, 0)));
    setEndAt((current) => (current === 0 || current > length ? length : current));
  }

  function moveStart(value) {
    const next = Math.min(value, Math.max(endAt - 0.1, 0));
    setStartAt(next);
    if (previewRef.current) previewRef.current.currentTime = next;
  }

  async function applyTrim() {
    if (!clip || endAt - startAt < 0.2) {
      setError("Choose a longer part of the video.");
      return;
    }
    setTrimming(true);
    setError(null);
    try {
      const trimmed = await trimFile(clip, startAt, endAt);
      const length = endAt - startAt;
      setDuration(length);
      setStartAt(0);
      setEndAt(length);
      setClip(trimmed);
    } catch (err) {
      setError(err.message);
    } finally {
      setTrimming(false);
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (!clip) {
      setError("Record a shot or choose a video.");
      return;
    }
    setSubmitting(true);
    setError(null);
    const body = new FormData();
    body.append("video", clip);
    try {
      const session = await api("/api/sessions", token, { method: "POST", body });
      onStarted(session);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page">
      <section className="panel">
        <h1 className="accent">Upload a jump shot</h1>
        <ul className="reminders">
          <li>
            <span className="mark" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <rect x="8" y="2.5" width="8" height="19" rx="2" fill="none" stroke="currentColor" strokeWidth="1.8" />
                <path d="M10.5 18.5h3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              </svg>
            </span>
            <span>Hold the phone upright.</span>
          </li>
          <li>
            <span className="mark" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <rect x="3" y="3" width="18" height="18" rx="2" fill="none" stroke="currentColor" strokeWidth="1.6" />
                <circle cx="12" cy="8" r="1.7" fill="none" stroke="currentColor" strokeWidth="1.6" />
                <path d="M12 10.2v4.2M9.2 12.6l2.8 1.2 2.8-1M9.6 18.2l2.4-3.6 2.4 3.6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <span>Side view, chest height, whole body in frame.</span>
          </li>
          <li>
            <span className="mark" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <circle cx="12" cy="5" r="1.8" fill="none" stroke="currentColor" strokeWidth="1.6" />
                <path d="M12 7.2v6.2M8.8 18.6l3.2-5 3.2 5M12 10.2l5.2-3.4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <span>One jump shot, then keep the arm up.</span>
          </li>
        </ul>
        <p className="hint">Open Video guide for the full steps and the example.</p>
      </section>
      <section className="panel">
        <video
          ref={previewRef}
          className="guide-video"
          muted={recording}
          playsInline
          controls={!recording}
          hidden={!recording && !clipUrl}
          src={recording ? undefined : clipUrl || undefined}
          onLoadedMetadata={noteLength}
          onDurationChange={noteLength}
        />
        <div className="record-row">
          {recording ? (
            <button type="button" className="stop" onClick={stopRecording}>Stop</button>
          ) : (
            <button type="button" className="record" onClick={startRecording} disabled={trimming}>Record</button>
          )}
          {clip && !recording ? <span className="hint">{clip.name}</span> : null}
        </div>
        {clip && !recording && duration > 0 ? (
          <div className="trim">
            <p className="hint">Start {startAt.toFixed(1)}s · End {endAt.toFixed(1)}s</p>
            <div className="trim-bar">
              <span
                className="trim-span"
                style={{
                  left: `${(startAt / duration) * 100}%`,
                  width: `${((endAt - startAt) / duration) * 100}%`,
                }}
              />
              <input
                type="range"
                min="0"
                max={duration}
                step="0.1"
                value={startAt}
                aria-label="Trim start"
                style={{ zIndex: startAt > endAt - 0.4 ? 5 : 3 }}
                onChange={(event) => moveStart(Number(event.target.value))}
              />
              <input
                type="range"
                min="0"
                max={duration}
                step="0.1"
                value={endAt}
                aria-label="Trim end"
                onChange={(event) => setEndAt(Math.max(Number(event.target.value), startAt + 0.1))}
              />
            </div>
            <div className="record-row">
              <button type="button" onClick={applyTrim} disabled={trimming}>
                {trimming ? "Trimming…" : "Trim"}
              </button>
              <button type="button" className="danger" onClick={deleteClip}>Delete</button>
            </div>
          </div>
        ) : null}
        {clip && !recording && duration === 0 ? (
          <div className="record-row">
            <button type="button" className="danger" onClick={deleteClip}>Delete</button>
          </div>
        ) : null}
        <form onSubmit={submit}>
          <label className="field">
            Or choose a video
            <input
              ref={fileRef}
              name="video"
              type="file"
              accept="video/mp4,video/quicktime,video/webm"
              disabled={recording || trimming}
              onChange={chooseFile}
            />
          </label>
          {error ? <p className="alert" role="alert">{error}</p> : null}
          <button type="submit" disabled={submitting || recording || trimming || !clip}>
            {submitting ? "Sending…" : "Start analysis"}
          </button>
        </form>
      </section>
    </div>
  );
}

function useVideoSrc(token, path) {
  const [src, setSrc] = useState(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    if (!token || !path) return undefined;
    let cancelled = false;
    let objectUrl;
    fetch(`${API}${path}`, { headers: { Authorization: `Bearer ${token}` } })
      .then(async (response) => {
        if (!response.ok) throw new Error(await readError(response));
        return response.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
        setMissing(false);
      })
      .catch(() => {
        if (!cancelled) setMissing(true);
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [token, path]);

  return { src, missing };
}

function ShotFilm({ token, path }) {
  const { src, missing } = useVideoSrc(token, path);
  return (
    <div className="shot-video">
      <h2>Your video</h2>
      {src ? (
        <video className="media-frame" src={src} controls playsInline />
      ) : (
        <div className="media-frame">
          <p>{missing ? "This video is missing." : "Loading video…"}</p>
        </div>
      )}
    </div>
  );
}

function Comparison({ session, token }) {
  const result = session.result;

  if (!result) return null;

  return (
    <div className="viewer">
      <section className="stage compare-stage">
        <ShotFilm token={token} path={`/api/sessions/${session.session_id}/video`} />
        <MeshPane />
      </section>
      <section className="report">
        <p className="eyebrow">Coaching analysis</p>
        <h1>Shot comparison</h1>
        <p className="hint">{session.video_name}</p>
        <article className="panel">
          <h2>Coaching tips</h2>
          <p>This section lists one tip for each difference. The tips cover:</p>
          {result.tips.every((tip) => tip.source === "placeholder") ? (
            <ul>
              <li>Elbow bend — how straight the shooting arm should be.</li>
              <li>Release height — where the wrist should be when the ball leaves the hand.</li>
              <li>Follow-through — holding the arm up after the shot.</li>
            </ul>
          ) : (
            <ul>
              {result.tips.map((tip) => (
                <li key={tip.text}>{tip.text}</li>
              ))}
            </ul>
          )}
        </article>
        <article className="panel">
          <h2>Shot measurements</h2>
          <p className="hint">Each measurement will show three figures: yours, the professional's, and the difference. The number on each row is an example of yours.</p>
          {result.features.map((feature) => (
            <p key={feature.name}>
              <strong>{FEATURE_LABELS[feature.name] || feature.name}:</strong> {formatFeature(feature)}
              {feature.name === "mean_elbow_angle" ? (
                <span className="hint">Bend of the shooting arm during the shot.</span>
              ) : null}
              {feature.name === "wrist_height_range" ? (
                <span className="hint">How far the wrist moves from the start of the shot to the release.</span>
              ) : null}
            </p>
          ))}
        </article>
      </section>
    </div>
  );
}

function AnalysisTab({ token, session }) {
  if (!session) {
    return (
      <div className="viewer">
        <section className="stage compare-stage">
          <div className="shot-video">
            <h2>How to record</h2>
            <video className="media-frame" src={GUIDE_VIDEO} controls playsInline />
          </div>
          <MeshPane />
        </section>
        <section className="report">
          <h1>How the app works</h1>
          <ol className="steps">
            <li><span className="num" aria-hidden="true">1</span><span>Video guide plays the recording clip.</span></li>
            <li><span className="num" aria-hidden="true">2</span><span>Upload records a shot or chooses a file.</span></li>
            <li><span className="num" aria-hidden="true">3</span><span>Analysis plays that video beside the mesh.</span></li>
            <li><span className="num" aria-hidden="true">4</span><span>History keeps the run.</span></li>
            <li><span className="num" aria-hidden="true">5</span><span>Profile holds the account and log out.</span></li>
          </ol>
        </section>
      </div>
    );
  }
  if (session.status === "failed") {
    return <div className="page"><h1>Analysis</h1><p className="alert" role="alert">{session.error}</p></div>;
  }
  if (session.status !== "ready") {
    return (
      <div className="viewer">
        <section className="stage compare-stage">
          <ShotFilm token={token} path={`/api/sessions/${session.session_id}/video`} />
          <MeshPane />
        </section>
        <section className="report">
          <h1>Analysis</h1>
          <p>{session.video_name}</p>
          <Progress status={session.status} />
        </section>
      </div>
    );
  }
  return <Comparison key={session.session_id} session={session} token={token} />;
}

function HistoryTab({ token, onOpen, onDeleted }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api("/api/sessions", token)
      .then((body) => { if (!cancelled) setRows(body); })
      .catch((err) => { if (!cancelled) setError(err.message); });
    return () => { cancelled = true; };
  }, [token]);

  async function remove(sessionId, videoName) {
    if (!window.confirm(`Delete ${videoName}?`)) return;
    try {
      await api(`/api/sessions/${sessionId}`, token, { method: "DELETE" });
      setRows((current) => current.filter((row) => row.session_id !== sessionId));
      onDeleted(sessionId);
    } catch (err) {
      setError(err.message);
    }
  }

  if (error) return <div className="page"><p className="alert" role="alert">{error}</p></div>;
  if (!rows) return <div className="page"><p>Loading history…</p></div>;
  return (
    <div className="page">
      <section className="panel">
        <h1>History</h1>
        {rows.length === 0 ? <p>No comparisons yet. Record a jump shot to add one.</p> : null}
        <ul className="history">
          {rows.map((row) => (
            <li key={row.session_id}>
              <button type="button" className="secondary" onClick={() => onOpen(row.session_id)}>
                <strong>{row.video_name}</strong>
                <small>{new Date(row.created_at).toLocaleString()} · {row.status}</small>
              </button>
              <button type="button" className="danger" onClick={() => remove(row.session_id, row.video_name)}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function ProfileTab({ token, user, onUser, onDeleted }) {
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);

  async function save(event) {
    event.preventDefault();
    setError(null);
    setSaved(false);
    const form = new FormData(event.currentTarget);
    try {
      const next = await api("/api/me", token, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          first_name: form.get("first_name"),
          last_name: form.get("last_name"),
          age: Number(form.get("age")),
          height_cm: Number(form.get("height_cm")),
          weight_kg: Number(form.get("weight_kg")),
        }),
      });
      onUser(next);
      setSaved(true);
    } catch (err) {
      setError(err.message);
    }
  }

  async function remove() {
    if (!window.confirm("Delete this account and every kept video and result?")) return;
    try {
      await api("/api/me", token, { method: "DELETE" });
      onDeleted();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="page">
      <section className="panel">
        <div className="profile-head">
          <div className="avatar" aria-hidden="true">{`${user.first_name[0] || ""}${user.last_name[0] || ""}`}</div>
          <div>
            <h1>{user.first_name} {user.last_name}</h1>
            <p className="hint">{user.email}</p>
          </div>
        </div>
        <form onSubmit={save}>
          <label className="field">First name<input name="first_name" defaultValue={user.first_name} required /></label>
          <label className="field">Last name<input name="last_name" defaultValue={user.last_name} required /></label>
          <label className="field">Age<input name="age" type="number" min="1" max="120" defaultValue={user.age} required /></label>
          <label className="field">Height (cm)<input name="height_cm" type="number" min="1" step="0.1" defaultValue={user.height_cm} required /></label>
          <label className="field">Weight (kg)<input name="weight_kg" type="number" min="1" step="0.1" defaultValue={user.weight_kg} required /></label>
          {error ? <p className="alert" role="alert">{error}</p> : null}
          {saved ? <p className="hint">Saved.</p> : null}
          <button type="submit">Save profile</button>
        </form>
        <div className="account-actions">
          <button type="button" className="logout" onClick={onDeleted}>Log out</button>
          <button type="button" className="danger" onClick={remove}>Delete account</button>
        </div>
      </section>
    </div>
  );
}

function App() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState(null);
  const [authError, setAuthError] = useState(null);
  const [tab, setTab] = useState("upload");
  const [session, setSession] = useState(null);

  useEffect(() => {
    if (!token) return undefined;
    let cancelled = false;
    api("/api/me", token)
      .then((body) => { if (!cancelled) setUser(body); })
      .catch((err) => {
        if (cancelled) return;
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
        setAuthError(err.message);
      });
    return () => { cancelled = true; };
  }, [token]);

  useEffect(() => {
    if (!token || !session?.session_id) return undefined;
    if (session.status === "ready" || session.status === "failed") return undefined;
    let cancelled = false;
    const timer = setInterval(() => {
      api(`/api/sessions/${session.session_id}`, token)
        .then((next) => { if (!cancelled) setSession(next); })
        .catch((err) => {
          if (!cancelled) setSession((current) => ({ ...current, status: "failed", error: err.message }));
        });
    }, 700);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [token, session?.session_id, session?.status]);

  function signIn(accessToken, nextUser) {
    localStorage.setItem(TOKEN_KEY, accessToken);
    setToken(accessToken);
    setUser(nextUser);
    setAuthError(null);
    setTab("upload");
  }

  function signOut() {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
    setSession(null);
    setTab("upload");
  }

  if (!token) return <AuthScreen onAuth={signIn} />;
  if (!user) {
    return (
      <main className="setup-screen">
        <section className="card">
          <span className="brand brand-mark"><BasketballLogo /></span>
          <h1>CourtSense</h1>
          <p>{authError || "Checking your account…"}</p>
        </section>
      </main>
    );
  }

  return (
    <div className="shell">
      <header className="nav">
        <Brand />
        {[["guide", "Video guide"], ["upload", "Upload"], ["analysis", "Analysis"], ["history", "History"]].map(([name, label]) => (
          <button key={name} type="button" className={tab === name ? "active" : ""} onClick={() => setTab(name)}>
            {label}
          </button>
        ))}
        <div className="nav-end">
          <button
            type="button"
            className={tab === "profile" ? "profile-link active" : "profile-link"}
            aria-label="Profile"
            onClick={() => setTab("profile")}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="12" cy="8" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.8" />
              <path d="M5.5 19.2c1.4-3 3.6-4.4 6.5-4.4s5.1 1.4 6.5 4.4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </header>
      {tab === "guide" ? <GuideTab /> : null}
      {tab === "upload" ? <UploadTab token={token} onStarted={(next) => { setSession(next); setTab("analysis"); }} /> : null}
      {tab === "analysis" ? <AnalysisTab token={token} session={session} /> : null}
      {tab === "history" ? (
        <HistoryTab
          token={token}
          onOpen={async (sessionId) => {
            const next = await api(`/api/sessions/${sessionId}`, token);
            setSession(next);
            setTab("analysis");
          }}
          onDeleted={(sessionId) => {
            setSession((current) => (current?.session_id === sessionId ? null : current));
          }}
        />
      ) : null}
      {tab === "profile" ? <ProfileTab token={token} user={user} onUser={setUser} onDeleted={signOut} /> : null}
    </div>
  );
}

export default App;
