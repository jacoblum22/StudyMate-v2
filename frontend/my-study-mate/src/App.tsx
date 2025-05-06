import { useState } from "react";

type TOCEntry = {
  heading: string;
  subtopics: string[];
};

function App() {
  const [toc, setToc] = useState<TOCEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("📚 MyStudyMate v2");

  const generateTOC = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/generate-toc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: "This is a fake transcript to generate a fake TOC.",
        }),
      });
      const data = await res.json();
      setToc(data);
    } catch (err) {
      setMessage("⚠️ Failed to generate TOC");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>{message}</h1>
      <button onClick={generateTOC} disabled={loading}>
        {loading ? "Generating..." : "Generate TOC"}
      </button>

      <ul style={{ marginTop: "2rem" }}>
        {toc.map((entry, i) => (
          <li key={i}>
            <strong>{entry.heading}</strong>
            <ul>
              {entry.subtopics.map((sub, j) => (
                <li key={j}>{sub}</li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
