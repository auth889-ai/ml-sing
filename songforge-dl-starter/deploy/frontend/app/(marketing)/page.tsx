import Link from "next/link";

// Landing page. A visitor arriving cold needs to know what this makes, why it
// is different, and how to try it — before any control surface appears.

const U = "https://images.unsplash.com/photo-";
const Q = "?auto=format&fit=crop&w=1100&q=80";

const CAPABILITIES = [
  {
    img: `${U}1520523839897-bd0b52f945a0${Q}`,
    alt: "Musicians performing on a lit stage",
    title: "Full arrangements",
    body: "Piano, guitar, bass, drums, strings, keys, brass and woodwinds — arranged together, not stacked loops.",
  },
  {
    img: `${U}1466428996289-fb355538da1b${Q}`,
    alt: "Close view of piano keys",
    title: "Acoustic realism",
    body: "Trained on real studio recordings of piano, violin and cello, so acoustic parts sound played rather than sampled.",
  },
  {
    img: `${U}1571330735066-03aaa9429d89${Q}`,
    alt: "Studio mixing console faders",
    title: "Finished, not raw",
    body: "Several takes are ranked, broken ones rejected, and the winner mastered to WAV and MP3.",
  },
];

const STEPS = [
  { h: "Describe it in plain words", p: "“piano intro building to a full-band climax” is a valid prompt. No genre menus to learn." },
  { h: "We plan the musical detail", p: "Genre, instruments, BPM, key, structure and energy are inferred — and anything you set yourself always wins." },
  { h: "Listen, download, keep", p: "Play it in the browser, download WAV or MP3, and it is saved to your library automatically." },
];

export default function LandingPage() {
  return (
    <>
      <section className="hero">
        <span className="eyebrow">Text to full song</span>
        <h1>
          Describe a song. Get a <span className="accent-text">finished track</span>.
        </h1>
        <p className="lede">
          SongForge turns a sentence into a complete, mastered piece of music — arranged,
          ranked across takes, checked for originality and exported ready to use.
        </p>
        <div className="hero-cta">
          <Link href="/create" className="btn btn-lg">Create a song — free</Link>
          <Link href="/how-it-works" className="btn btn-ghost btn-lg">See how it works</Link>
        </div>

        <div className="hero-shot">
          <img
            src={`${U}1511671782779-c97d3d27a1d4?auto=format&fit=crop&w=1800&q=80`}
            alt="Studio headphones resting on a mixing desk"
            loading="eager"
          />
          <div className="veil" />
        </div>
        <p className="credit">Photography: Unsplash</p>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Built for whole songs, not clips</h2>
          <p>
            Six capability families were assembled specifically to fix what a single-corpus
            model gets wrong.
          </p>
        </div>
        <div className="grid">
          {CAPABILITIES.map((c) => (
            <article className="tile" key={c.title}>
              <div className="tile-media">
                <img src={c.img} alt={c.alt} loading="lazy" />
              </div>
              <div className="tile-body">
                <h3>{c.title}</h3>
                <p>{c.body}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="split">
          <div>
            <span className="eyebrow">Why it is different</span>
            <h2 style={{ fontSize: "clamp(1.4rem, 2.6vw, 1.9rem)", marginTop: 8 }}>
              Music you can actually ship
            </h2>
            <p className="lede" style={{ marginTop: 10 }}>
              Most generators cannot tell you what they were trained on. Every second behind
              SongForge is CC0 or CC BY, verified against each source&rsquo;s own metadata. We
              censused 106,574 tracks and discarded 97,735 of them to keep 8,839 clean ones.
            </p>
            <p className="lede" style={{ marginTop: 10 }}>
              Generated audio is then screened against a fingerprint database, so a track that
              leans too close to its training data gets flagged before you ever hear it.
            </p>
            <div className="hero-cta" style={{ justifyContent: "flex-start", marginTop: 20 }}>
              <Link href="/research" className="btn btn-ghost">Read the numbers</Link>
            </div>
          </div>
          <div className="split-media">
            <img
              src={`${U}1470225620780-dba8ba36b745${Q}`}
              alt="Recording studio control room"
              loading="lazy"
            />
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Three steps</h2>
          <p>No account required to try it.</p>
        </div>
        <div className="steps">
          {STEPS.map((s) => (
            <div className="step" key={s.h}>
              <span className="step-n" aria-hidden="true" />
              <div>
                <h3>{s.h}</h3>
                <p>{s.p}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="cta-band">
        <h2>Make your first song</h2>
        <p>
          Type one sentence. Everything else — arrangement, ranking, mastering — happens for you.
        </p>
        <Link href="/create" className="btn btn-lg">Open the app</Link>
      </section>
    </>
  );
}
