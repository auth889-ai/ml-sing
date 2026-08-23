"use client";

// Persistent left rail: navigation plus the visitor's saved library.
//
// The library lives here rather than on its own page alone so a returning
// visitor sees their shelf immediately — the shelf is the evidence that this
// produced something, and burying it one click deep hides the product's proof.

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { listSongs, formatWhen, type SavedSong } from "../lib/library";

const NAV = [
  { href: "/create", icon: "✦", label: "Create" },
  { href: "/library", icon: "◍", label: "Library" },
  
  { href: "/", icon: "←", label: "Back to site" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [songs, setSongs] = useState<SavedSong[]>([]);

  useEffect(() => {
    const sync = () => setSongs(listSongs());
    sync();
    // Same-tab writes dispatch this; storage covers other tabs.
    window.addEventListener("songforge:library", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("songforge:library", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  return (
    <aside className="sidebar">
      <Link href="/" className="brand">
        <span className="brand-mark">♪</span>
        <span className="brand-name">SongForge</span>
      </Link>

      <nav className="nav">
        {NAV.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="nav-link"
            aria-current={pathname === item.href ? "page" : undefined}
          >
            <span className="nav-ico" aria-hidden="true">{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="side-label">Saved library</div>

      <div className="library-box">
        <div className="library-head">
          <h3 style={{ fontSize: "0.82rem" }}>Your songs</h3>
          {songs.length > 0 && <span className="pill">{songs.length}</span>}
        </div>

        {songs.length === 0 ? (
          <p className="library-empty">
            Nothing saved yet. Songs you generate are kept here, in this browser.
          </p>
        ) : (
          <div className="library-list">
            {songs.slice(0, 8).map((song) => (
              <Link key={song.id} href={`/library#${song.id}`} className="library-item">
                <div className="t">{song.prompt || "Untitled"}</div>
                <div className="m">
                  {formatWhen(song.createdAt)}
                  {song.durationSeconds ? ` · ${song.durationSeconds}s` : ""}
                </div>
              </Link>
            ))}
          </div>
        )}

        {songs.length > 8 && (
          <Link href="/library" className="hint" style={{ paddingLeft: 4 }}>
            View all {songs.length} →
          </Link>
        )}
      </div>
    </aside>
  );
}
