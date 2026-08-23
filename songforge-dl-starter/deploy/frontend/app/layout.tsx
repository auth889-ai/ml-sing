import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "SongForge — text to full song",
  description:
    "Describe a song in plain words and get a finished, downloadable track. Built on the pretrained ACE-Step 1.5 XL-turbo foundation (MIT) with a SongForge-trained adapter, on a corpus that is CC0/CC-BY throughout.",
};

// Root layout carries only the document. The marketing surface and the app
// each own their own chrome — a landing page with a sidebar reads as an
// internal tool, and an app with a marketing nav wastes the width.
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
