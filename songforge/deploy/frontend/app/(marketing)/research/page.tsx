// Data & licensing — the part most generators will not show you.
//
// Every figure here is a measured count from the corpus reports in the repo,
// not an estimate. If a number cannot be backed by a report it does not
// appear on this page.

const FAMILIES = [
  { family: "Arrangement", source: "Slakh2100-redux", licence: "CC BY 4.0", role: "Multi-instrument interaction, stem balance" },
  { family: "Real songs", source: "Free Music Archive", licence: "CC0 / CC BY, per track", role: "Production realism, genre diversity" },
  { family: "Vocals", source: "VocalSet", licence: "CC BY 4.0", role: "Singing realism, phrasing" },
  { family: "Piano, violin, cello, strings", source: "MusicNet", licence: "CC BY 4.0", role: "Acoustic and cinematic realism" },
  { family: "Guitar", source: "GuitarSet", licence: "CC BY 4.0", role: "Timbre, articulation, voicing" },
  { family: "Drums", source: "Slakh isolated drum stems", licence: "CC BY 4.0", role: "Percussion" },
];

const GATES = [
  "licence", "provenance", "cross-corpus duplicate", "integrity / decode",
  "silence", "clipping / quality", "split leakage", "metadata",
  "rich caption", "deployability",
];

export default function DataPage() {
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">Data &amp; licensing</span>
        <h1>
          Every training second is <span className="accent-text">CC0 or CC BY</span>
        </h1>
        <p className="lede">
          Verified against each source&rsquo;s own metadata, not assumed. NonCommercial,
          NoDerivatives, ShareAlike and unresolved records never enter the deployable model.
        </p>
      </header>

      <section className="card">
        <h2 className="card-title">The Free Music Archive census</h2>
        <p style={{ color: "var(--text-2)", fontSize: "0.88rem", marginBottom: 12 }}>
          FMA ships 106,574 tracks whose licences differ track by track, and the maintainers
          state plainly that they do not own all audio rights. So we downloaded the 342 MB of
          metadata first, censused every row, and built the subset from that result.
        </p>
        <dl className="kv">
          <dt>Tracks censused</dt><dd>106,574</dd>
          <dt>CC0 / public domain</dt><dd>1,820</dd>
          <dt>CC BY</dt><dd>7,019</dd>
          <dt><strong>Deployable</strong></dt><dd><strong>8,839 (606.1 hours)</strong></dd>
          <dt>Excluded — NonCommercial</dt><dd>93,713</dd>
          <dt>Excluded — ShareAlike</dt><dd>2,802</dd>
          <dt>Excluded — NoDerivatives</dt><dd>903</dd>
          <dt>Excluded — unresolved</dt><dd>317</dd>
        </dl>
        <div className="note note-ok" style={{ marginTop: 12 }}>
          97,735 of 106,574 tracks were discarded to keep 8,839 clean ones — 8.3% of the
          catalogue survives the licence gate.
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Six capability families</h2>
        <p style={{ color: "var(--text-2)", fontSize: "0.88rem", marginBottom: 12 }}>
          Weighted at sampling time, never concatenated — otherwise the largest corpus drowns
          the small ones that exist precisely to fix known weaknesses.
        </p>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.84rem" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                <th style={{ padding: "6px 8px" }}>Family</th>
                <th style={{ padding: "6px 8px" }}>Source</th>
                <th style={{ padding: "6px 8px" }}>Licence</th>
                <th style={{ padding: "6px 8px" }}>Role</th>
              </tr>
            </thead>
            <tbody>
              {FAMILIES.map((f) => (
                <tr key={f.family} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}><strong>{f.family}</strong></td>
                  <td style={{ padding: "8px" }}>{f.source}</td>
                  <td style={{ padding: "8px" }}>{f.licence}</td>
                  <td style={{ padding: "8px", color: "var(--text-2)" }}>{f.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Ten gates, in order</h2>
        <p style={{ color: "var(--text-2)", fontSize: "0.88rem", marginBottom: 12 }}>
          Every record passes all ten. A record failing any gate is excluded and counted in
          the corpus report. No gate is skipped for speed.
        </p>
        <div className="dl-row">
          {GATES.map((g, i) => (
            <span className="dl" key={g} style={{ cursor: "default" }}>
              {i + 1}. {g}
            </span>
          ))}
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">Training result</h2>
        <dl className="kv">
          <dt>Trainable parameters</dt><dd>1,835,008 of 4,991,023,206 (0.04%)</dd>
          <dt>Optimizer steps / epoch</dt><dd>907</dd>
          <dt>Loss by epoch</dt><dd>0.9691 → 0.9228 → 0.8809</dd>
          <dt>Gradient coverage</dt><dd>768 / 768 tensors non-zero and finite</dd>
        </dl>
        <p className="hint" style={{ marginTop: 10 }}>
          Gradient coverage is read from AdamW&rsquo;s exp_avg_sq — the running mean of squared
          gradients — so a non-zero value is direct evidence a parameter received real gradient
          signal, not an inference from a falling loss curve.
        </p>
      </section>
    </>
  );
}
