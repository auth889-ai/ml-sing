import type { ReactNode } from "react";

export const metadata = {
  title: "SongForge",
  description:
    "Generate a complete song from a text prompt. Built on the pretrained ACE-Step 1.5 XL-turbo model (MIT) with a SongForge-trained adapter.",
};

// One inline stylesheet, no CSS framework. UI polish is the last priority in
// the 100-hour plan; this needs to be readable, honest, and nothing more.
const css = `
  * { box-sizing: border-box; }
  body { margin: 0; background: #0f1115; color: #e6e6e6;
         font: 16px/1.5 system-ui, -apple-system, sans-serif; }
  main { max-width: 720px; margin: 0 auto; padding: 2rem 1rem 4rem; }
  h1 { font-size: 1.6rem; margin: 0 0 0.25rem; }
  .sub { color: #9aa0ac; margin: 0 0 1.5rem; font-size: 0.9rem; }
  label { display: block; margin: 0.9rem 0 0.25rem; font-size: 0.85rem; color: #c3c8d2; }
  input, textarea, select {
    width: 100%; padding: 0.5rem 0.6rem; border-radius: 6px;
    border: 1px solid #2a2f3a; background: #171a21; color: #e6e6e6; font: inherit;
  }
  textarea { resize: vertical; }
  .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0 0.75rem; }
  button {
    margin-top: 1.25rem; padding: 0.6rem 1.4rem; border-radius: 6px; border: 0;
    background: #4f7cff; color: white; font: inherit; font-weight: 600; cursor: pointer;
  }
  button:disabled { background: #2a2f3a; color: #777; cursor: default; }
  .status { margin-top: 1rem; color: #9aa0ac; }
  .warn { margin-top: 0.75rem; padding: 0.6rem 0.8rem; border-radius: 6px;
          background: #2b2313; color: #e8c76a; font-size: 0.85rem; }
  .error { margin-top: 0.75rem; padding: 0.6rem 0.8rem; border-radius: 6px;
           background: #2b1515; color: #ef8a8a; font-size: 0.9rem; }
  .player { margin-top: 1.25rem; }
  audio { width: 100%; }
  a.download { color: #7ea2ff; font-size: 0.9rem; }
  .hint { color: #6b7280; font-size: 0.78rem; margin-top: 0.15rem; }
  footer { margin-top: 3rem; color: #6b7280; font-size: 0.78rem; border-top: 1px solid #2a2f3a; padding-top: 0.75rem; }
`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <style dangerouslySetInnerHTML={{ __html: css }} />
      </head>
      <body>
        <main>{children}</main>
      </body>
    </html>
  );
}
