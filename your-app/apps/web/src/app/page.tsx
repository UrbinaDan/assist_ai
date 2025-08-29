"use client";
import { useState } from "react";
import { useAudioStream } from "@/hooks/useAudioStream";

export default function Home() {
  const [mode, setMode] = useState<"default" | "interview">("default");
  const { enabled, level, start, stop } = useAudioStream();

  return (
    <main className="min-h-screen flex flex-col items-center justify-between p-4 max-w-xl mx-auto">
      <div className="w-full">
        <h1 className="text-3xl font-bold mb-2">Assist</h1>
        <div className="flex items-center gap-2 mb-4">
          <button
            className={`px-4 py-2 rounded ${enabled ? "bg-red-600" : "bg-green-600"} text-white font-semibold`}
            onClick={enabled ? stop : start}
          >
            {enabled ? "Stop Recording" : "Start Recording"}
          </button>
          <select
            className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1"
            value={mode}
            onChange={e => setMode(e.target.value as "default" | "interview")}
          >
            <option value="default">Default</option>
            <option value="interview">Interview</option>
          </select>
        </div>
        <div className="w-full h-3 bg-zinc-800 rounded mb-6 overflow-hidden">
          <div
            className="h-full bg-green-500 transition-all"
            style={{ width: `${Math.min(1, level) * 100}%` }}
          />
        </div>
        <section className="mb-6">
          <h2 className="text-xl font-semibold mb-1">Transcript</h2>
          <div className="bg-zinc-900 rounded p-3 min-h-[60px] text-zinc-300">
            {/* TODO: Show live transcript from ASR WebSocket */}
            <span className="italic text-zinc-500">Transcript will appear here…</span>
          </div>
        </section>
        {mode === "interview" && (
          <section className="mb-6">
            <h2 className="text-xl font-semibold mb-1">Interview Tips</h2>
            <ul className="bg-zinc-900 rounded p-3 text-zinc-300 list-disc pl-5">
              {/* TODO: Show live interview tips from WebSocket */}
              <li>Keep your answers concise.</li>
              <li>Maintain eye contact.</li>
              <li>Highlight your achievements.</li>
            </ul>
          </section>
        )}
      </div>
      <footer className="w-full text-center text-xs text-zinc-500 mt-8">
        PWA-ready. Use ‘Add to Home Screen’ on mobile to install.
      </footer>
    </main>
  );
}
