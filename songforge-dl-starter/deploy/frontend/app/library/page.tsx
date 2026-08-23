"use client";

// The saved library as a full page: every song the visitor has generated in
// this browser, playable inline.
//
// Audio lives on the GPU host and is retained for a few hours, so a saved row
// can outlive its file. The page says so on each item rather than letting a
// silent 404 look like a bug in the product.

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  listSongs,
  removeSong,
  clearLibrary,
  audioUrlFor,
  formatWhen,
  type SavedSong,
} from "../../lib/library";

export default function LibraryPage() {
  const [songs, setSongs] = useState<SavedSong[]>([]);

  useEffect(() => {
    const sync = () => setSongs(listSongs());
    sync();
    window.addEventListener("songforge:library", sync);
    return () => window.removeEventListener("songforge:library", sync);
  }, []);

  return (
    <>
      <header className="page-head">
        <span className="eyebrow">Library</span>
        <h1>Your saved songs</h1>
        <p className="lede">
          Kept in this browser only — no account, no server-side profile. Clearing site data
          clears this shelf.
        </p>
      </header>

      {songs.length === 0 ? (
        <section className="card">
          <h2 className="card-title">Nothing here yet</h2>
          <p style={{ color: "var(--text-2)" }}>
            Generate a song and it will appear here automatically.
          </p>
          <div style={{ marginTop: 14 }}>
            <Link href="/" className="btn">Create a song</Link>
          </div>
        </section>
      ) : (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span className="hint">{songs.length} saved</span>
            <button
              className="btn-ghost"
              style={{ padding: "6px 12px", fontSize: "0.8rem" }}
              onClick={() => {
                if (confirm("Remove every saved song from this browser?")) clearLibrary();
              }}
            >
              Clear library
            </button>
          </div>

          {songs.map((song) => (
            <section className="card" key={song.id} id={song.id}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                <h2 className="card-title" style={{ marginBottom: 4 }}>
                  {song.prompt || "Untitled"}
                </h2>
                <button
                  className="btn-ghost"
                  style={{ padding: "4px 10px", fontSize: "0.75rem", height: "fit-content" }}
                  onClick={() => removeSong(song.id)}
                  aria-label="Remove from library"
                >
                  Remove
                </button>
              </div>

              <dl className="kv" style={{ marginTop: 8 }}>
                <dt>Created</dt>
                <dd>{formatWhen(song.createdAt)}</dd>
                {song.durationSeconds != null && (
                  <>
                    <dt>Duration</dt>
                    <dd>{song.durationSeconds}s</dd>
                  </>
                )}
                {song.seed != null && (
                  <>
                    <dt>Seed</dt>
                    <dd>{song.seed}</dd>
                  </>
                )}
              </dl>

              <audio controls src={audioUrlFor(song)} />
              <div className="dl-row">
                <a className="dl" href={audioUrlFor(song)} download={`songforge_${song.jobId}.wav`}>
                  ↓ Download WAV
                </a>
              </div>
              <p className="hint">
                Audio is retained on the server for a few hours after rendering; older entries
                may no longer play.
              </p>
            </section>
          ))}
        </>
      )}
    </>
  );
}
