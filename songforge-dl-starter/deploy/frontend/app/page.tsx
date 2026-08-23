"use client";

// Create page: prompt → submit → poll → play → download → save to library.
//
// Control labels come from /api/capabilities so no knob is presented as
// stronger than it is: "native" controls condition the model, "prompt"
// controls are only text the model may or may not follow. A knob that
// silently does nothing is worse than a missing knob.

import { useEffect, useRef, useState } from "react";
import { capabilities, submit, waitFor, type GenerateBody, type JobState } from "../lib/api";
import { saveSong } from "../lib/library";

interface Limits {
  min_duration_seconds: number;
  max_duration_seconds: number;
  max_prompt_chars: number;
  max_lyrics_chars: number;
}

const IDEAS = [
  "grand piano solo, slow and cinematic",
  "violin and cello duet, minor key, mournful",
  "acoustic guitar and soft female vocal",
  "piano intro building to a full-band climax",
];

export default function Page() {
  const [prompt, setPrompt] = useState("");
  const [lyrics, setLyrics] = useState("");
  const [duration, setDuration] = useState("60");
  const [bpm, setBpm] = useState("");
  const [key, setKey] = useState("");
  const [timeSignature, setTimeSignature] = useState("");
  const [language, setLanguage] = useState("en");
  const [seed, setSeed] = useState("0");
  const [advanced, setAdvanced] = useState(false);

  const [limits, setLimits] = useState<Limits | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const jobIdRef = useRef<string | null>(null);

  useEffect(() => {
    capabilities()
      .then((c) => setLimits(c.limits as Limits))
      .catch(() => setLimits(null)); // backend warming up; submit still validates server-side
  }, []);

  async function onGenerate() {
    setError(null);
    setJob(null);
    const body: GenerateBody = {
      prompt: prompt.trim(),
      seed: Number(seed) || 0,
      duration_seconds: Number(duration) || undefined,
      vocal_language: language || undefined,
    };
    if (lyrics.trim()) body.lyrics = lyrics.trim();
    if (bpm) body.bpm = Number(bpm);
    if (key.trim()) body.key = key.trim();
    if (timeSignature.trim()) body.time_signature = timeSignature.trim();

    setBusy(true);
    try {
      const submitted = await submit(body);
      jobIdRef.current = submitted.job_id;
      setJob(submitted);
      const settled = await waitFor(submitted.job_id, (s) => {
        if (jobIdRef.current === submitted.job_id) setJob(s);
      });
      if (jobIdRef.current === submitted.job_id) {
        setJob(settled);
        if (settled.status === "done") {
          saveSong({
            id: settled.job_id,
            jobId: settled.job_id,
            prompt: body.prompt,
            createdAt: Date.now(),
            durationSeconds: body.duration_seconds,
            seed: body.seed,
          });
        } else {
          setError(settled.error ?? `Job ${settled.status}`);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const audioUrl =
    job?.status === "done" && job.audio_url ? `/api/songforge/jobs/${job.job_id}/audio` : null;

  return (
    <>
      <header className="page-head">
        <span className="eyebrow">Create</span>
        <h1>
          Describe a song, get a <span className="accent-text">finished track</span>
        </h1>
        <p className="lede">
          A free-form prompt is planned into genre, instruments, key and structure, rendered
          by ACE-Step 1.5 with the SongForge adapter, ranked across takes, screened for
          originality, then mastered to WAV and MP3.
        </p>
      </header>

      <section className="card">
        <label htmlFor="prompt">Prompt — instruments, style, mood</label>
        <textarea
          id="prompt"
          rows={3}
          maxLength={limits?.max_prompt_chars}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="warm acoustic ballad, fingerpicked guitar, soft female vocal, gentle strings"
        />
        <div className="dl-row">
          {IDEAS.map((idea) => (
            <button
              key={idea}
              type="button"
              className="dl"
              style={{ cursor: "pointer" }}
              onClick={() => setPrompt(idea)}
            >
              {idea}
            </button>
          ))}
        </div>

        <label htmlFor="lyrics">Lyrics — optional, leave empty for instrumental</label>
        <textarea
          id="lyrics"
          rows={4}
          maxLength={limits?.max_lyrics_chars}
          value={lyrics}
          onChange={(e) => setLyrics(e.target.value)}
          placeholder={"[verse]\nYour lines here…"}
        />

        <div className="row">
          <div>
            <label htmlFor="duration">Duration (s)</label>
            <input
              id="duration"
              type="number"
              min={limits?.min_duration_seconds ?? 10}
              max={limits?.max_duration_seconds ?? 120}
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
            />
            {limits && (
              <div className="hint">
                {limits.min_duration_seconds}–{limits.max_duration_seconds}s on this deployment
              </div>
            )}
          </div>
          <div>
            <label htmlFor="seed">Seed</label>
            <input id="seed" type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
            <div className="hint">Same inputs + seed → same song</div>
          </div>
        </div>

        <button
          type="button"
          className="btn-ghost"
          style={{ marginTop: 14, padding: "7px 13px", fontSize: "0.82rem" }}
          onClick={() => setAdvanced((v) => !v)}
        >
          {advanced ? "Hide" : "Show"} advanced controls
        </button>

        {advanced && (
          <div className="row" style={{ marginTop: 12 }}>
            <div>
              <label htmlFor="bpm">BPM</label>
              <input id="bpm" type="number" min={30} max={300} value={bpm} placeholder="auto"
                onChange={(e) => setBpm(e.target.value)} />
            </div>
            <div>
              <label htmlFor="key">Key</label>
              <input id="key" value={key} placeholder="e.g. A minor"
                onChange={(e) => setKey(e.target.value)} />
            </div>
            <div>
              <label htmlFor="ts">Time signature</label>
              <input id="ts" value={timeSignature} placeholder="e.g. 4/4"
                onChange={(e) => setTimeSignature(e.target.value)} />
            </div>
            <div>
              <label htmlFor="lang">Vocal language</label>
              <input id="lang" value={language} onChange={(e) => setLanguage(e.target.value)} />
            </div>
          </div>
        )}

        <div style={{ marginTop: 18 }}>
          <button onClick={onGenerate} disabled={busy || !prompt.trim()}>
            {busy ? "Generating…" : "Generate song"}
          </button>
        </div>

        {job && (job.status === "queued" || job.status === "running") && (
          <div className="note note-info">
            {job.status === "queued"
              ? `Queued${job.queue_position != null ? ` — position ${job.queue_position}` : ""}…`
              : "Rendering on the GPU…"}
          </div>
        )}

        {job && job.control_warnings.length > 0 && (
          <div className="note note-warn">
            {job.control_warnings.map((w) => (
              <div key={w}>⚠ {w}</div>
            ))}
          </div>
        )}

        {error && <div className="note note-error">{error}</div>}

        {audioUrl && (
          <div>
            <div className="note note-ok">Done — saved to your library.</div>
            <audio controls src={audioUrl} />
            <div className="dl-row">
              <a className="dl" href={audioUrl} download={`songforge_${job!.job_id}.wav`}>
                ↓ Download WAV
              </a>
            </div>
          </div>
        )}
      </section>

      <footer>
        Foundation: ACE-Step 1.5 XL-turbo (MIT, pretrained — not ours). Adapter, planner,
        ranking, originality screen, finishing and this service: SongForge. Generated audio
        is deleted from the server a few hours after rendering.
      </footer>
    </>
  );
}
