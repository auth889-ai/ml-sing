import type { ReactNode } from "react";
import Sidebar from "../Sidebar";

// App chrome: persistent left rail with navigation and the saved library.
// Separate from the marketing layout so the working surface gets the full
// width and none of the landing-page furniture.
export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <div className="shell">
      <Sidebar />
      <main className="main">
        <div className="container">{children}</div>
      </main>
    </div>
  );
}
