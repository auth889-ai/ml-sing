// Privacy — what is stored, where, and what is not.
//
// Written to be checkable rather than reassuring. Every claim here maps to
// something visible in the source: the library lives in localStorage, the
// proxy route never forwards cookies, and the backend keeps audio on a TTL.

export default function PrivacyPage() {
  return (
    <>
      <section className="hero" style={{ paddingBottom: 12, textAlign: "left" }}>
        <span className="eyebrow">Privacy &amp; data</span>
        <h1 style={{ margin: "12px 0 0", maxWidth: "24ch" }}>
          What we store, and what we <span className="accent-text">deliberately don&rsquo;t</span>
        </h1>
        <p className="lede" style={{ margin: "14px 0 0" }}>
          SongForge has no user accounts. That is a design decision, not an omission — and it
          is why there is very little about you for anyone to lose.
        </p>
      </section>

      <div style={{ maxWidth: 760 }}>
        <section className="card">
          <h2 className="card-title">There is no login, and that is the point</h2>
          <p style={{ color: "var(--text-2)", fontSize: "0.9rem" }}>
            Your saved library lives in <strong>your own browser</strong>, in localStorage,
            under the key <code>songforge.library.v1</code>. It never reaches our servers. No
            email, no password, no profile, no analytics account.
          </p>
          <div className="note note-info">
            Because nothing is stored server-side, a login box would gate the interface without
            protecting any data — security theatre. If per-device sync is added later it will be
            opt-in, and this page will say exactly what changed.
          </div>
        </section>

        <section className="card">
          <h2 className="card-title">What each part actually holds</h2>
          <dl className="kv">
            <dt>Your browser</dt>
            <dd>Saved song list: prompt text, timestamp, duration, seed, job id. Clearing site data erases it.</dd>
            <dt>The GPU host</dt>
            <dd>Rendered audio, retained a few hours and then deleted, so links expire by design.</dd>
            <dt>This web app</dt>
            <dd>Proxies your request to the backend. It forwards your IP so the backend can rate-limit fairly, and nothing else.</dd>
            <dt>Never collected</dt>
            <dd>Email, password, payment details, location, third-party trackers.</dd>
          </dl>
        </section>

        <section className="card">
          <h2 className="card-title">Your prompts</h2>
          <p style={{ color: "var(--text-2)", fontSize: "0.9rem" }}>
            A prompt is sent to the backend to render your song and appears in the server log
            for that request. Do not put anything confidential in a prompt — the same advice
            applies to every hosted generation service, and most of them will not tell you.
          </p>
        </section>

        <section className="card">
          <h2 className="card-title">Rights in what you generate</h2>
          <p style={{ color: "var(--text-2)", fontSize: "0.9rem" }}>
            The model was adapted only on CC0 and CC BY audio, with NonCommercial,
            NoDerivatives and ShareAlike material excluded outright. Output is additionally
            screened against a corpus fingerprint database. That is a materially stronger
            position than a generator trained on unlicensed audio — but it is not legal advice,
            and it is not a guarantee about any individual output.
          </p>
        </section>

        <section className="card">
          <h2 className="card-title">Verifying any of this</h2>
          <p style={{ color: "var(--text-2)", fontSize: "0.9rem" }}>
            The whole system is open source. The library code, the proxy route and the
            retention setting are all readable — you do not have to take this page&rsquo;s word for it.
          </p>
          <div className="dl-row">
            <a className="dl" href="https://github.com/auth889-ai/ml-sing">Read the source</a>
          </div>
        </section>
      </div>
    </>
  );
}
