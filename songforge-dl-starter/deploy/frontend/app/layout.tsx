import type { ReactNode } from "react";
import "./globals.css";
import Sidebar from "./Sidebar";

export const metadata = {
  title: "SongForge — text to full song",
  description:
    "Generate a complete song from a free-form prompt. Built on the pretrained ACE-Step 1.5 XL-turbo foundation (MIT) with a SongForge-trained LoKr adapter, on a corpus that is CC0/CC-BY throughout.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Sidebar />
          <main className="main">
            <div className="container">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
