import type { ReactNode } from "react";
import Link from "next/link";

// Public chrome: top navigation and a footer. The technical pages —
// how it works, research, privacy — live here rather than in the app's
// sidebar, so a visitor deciding whether to try this can read them, and a
// user making a song is not shown training internals on the way.
export default function MarketingLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <header className="site-nav">
        <Link href="/" className="brand">
          <span className="brand-mark">♪</span>
          <span className="brand-name">SongForge</span>
        </Link>
        <nav className="links">
          <Link href="/how-it-works">How it works</Link>
          <Link href="/research">Research</Link>
          <Link href="/privacy">Privacy</Link>
          <Link href="/create" className="btn" style={{ padding: "8px 17px", fontSize: "0.86rem" }}>
            Open the app
          </Link>
        </nav>
      </header>

      <div className="marketing">{children}</div>

      <footer className="site-footer">
        <div className="footer-inner">
          <div className="footer-col">
            <strong>Product</strong>
            <Link href="/create">Create a song</Link>
            <Link href="/library">Your library</Link>
            <Link href="/how-it-works">How it works</Link>
          </div>
          <div className="footer-col">
            <strong>Transparency</strong>
            <Link href="/research">Research &amp; results</Link>
            <Link href="/privacy">Privacy &amp; data</Link>
            <a href="https://github.com/auth889-ai/ml-sing">Source code</a>
          </div>
          <div className="footer-col">
            <strong>Built on</strong>
            <span>ACE-Step 1.5 XL-turbo (MIT)</span>
            <span>Corpus: CC0 / CC BY only</span>
            <span>MIT licensed</span>
          </div>
        </div>
        <p className="footer-note">
          SongForge adapts a frozen open foundation model; it does not claim to have trained
          one from scratch. Photography from Unsplash under the Unsplash License.
        </p>
      </footer>
    </>
  );
}
