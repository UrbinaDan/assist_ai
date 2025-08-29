import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Assist",
  description: "Live transcription + interview coach",
  manifest: "/manifest.json",
  themeColor: "#0b0b0b",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/manifest.json" />
      </head>
      <body className="bg-black text-white">
        {children}
        <script
          dangerouslySetInnerHTML={{
            __html: `
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js').catch(() => {});
  });
}
`,
          }}
        />
      </body>
    </html>
  );
}
