import { useState, useRef } from "react";

const ACCENT = "#10b981";
const JD_DEFAULT = `Senior Data Scientist

Requirements:
- 3+ years of experience in machine learning and data science
- Proficiency in Python, pandas, NumPy, scikit-learn
- Experience with TensorFlow or PyTorch
- Strong SQL and data visualization skills (Matplotlib, Tableau)
- NLP experience with transformer models (BERT, GPT)
- Cloud platforms: AWS, GCP, or Azure
- MLOps: Docker, Kubernetes, MLflow
- Communication and teamwork skills
- MS/BS in Computer Science, Statistics, or related field`;

const RESUME_PRESETS = [
  {
    name: "Alice Johnson",
    text: `Senior Data Scientist with 5 years experience. Expert in Python, pandas, NumPy, scikit-learn, TensorFlow, and PyTorch. Built NLP pipelines using BERT and transformer models. Proficient in SQL, Tableau, and Matplotlib. Deployed models on AWS and GCP using Docker and Kubernetes. Used MLflow for experiment tracking. MS Computer Science. Strong communicator and team leader.`,
  },
  {
    name: "Bob Smith",
    text: `Data Analyst with 2 years experience. Skilled in Python, pandas, and basic scikit-learn. Familiar with SQL and Excel. Created Tableau dashboards. Some matplotlib usage. Learning TensorFlow. BS Statistics. Good team player.`,
  },
  {
    name: "Carol Lee",
    text: `ML Engineer with 4 years experience. Deep expertise in PyTorch, TensorFlow, scikit-learn. Production NLP with BERT and GPT. Strong Python and SQL. Models deployed on Azure and AWS using Docker. MLflow and model monitoring. MS Artificial Intelligence. Published transformer research.`,
  },
  {
    name: "David Park",
    text: `Full Stack Developer with 3 years experience. JavaScript, React, Node.js, PostgreSQL. Some Python scripting. Docker and Kubernetes for deployment. AWS for cloud hosting. BS Computer Engineering. No formal machine learning experience.`,
  },
  {
    name: "Eva Martinez",
    text: `Data Scientist with 3 years experience. Python, NumPy, pandas, scikit-learn, Seaborn. Built classification and regression models. NLP and text classification. SQL for data extraction. Some TensorFlow exposure. GCP data processing. BS Mathematics. Strong analytical and communication skills.`,
  },
];

function ScoreBar({ value, max = 100 }) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  const color =
    pct >= 70 ? "#10b981" : pct >= 50 ? "#3b82f6" : pct >= 30 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ width: "100%", background: "#1e293b", borderRadius: 6, height: 8, overflow: "hidden" }}>
      <div
        style={{
          width: `${pct}%`,
          height: "100%",
          background: color,
          borderRadius: 6,
          transition: "width 0.8s ease",
        }}
      />
    </div>
  );
}

function Chip({ label, type }) {
  const colors = {
    matched: { bg: "#052e16", text: "#4ade80", border: "#166534" },
    missing: { bg: "#2d0a0a", text: "#f87171", border: "#7f1d1d" },
    skill:   { bg: "#0f172a", text: "#94a3b8", border: "#1e293b" },
  };
  const c = colors[type] || colors.skill;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: 4,
        fontSize: 11,
        fontFamily: "monospace",
        background: c.bg,
        color: c.text,
        border: `1px solid ${c.border}`,
        marginRight: 4,
        marginBottom: 4,
      }}
    >
      {label}
    </span>
  );
}

function TierBadge({ score }) {
  if (score >= 70) return <span style={{ color: "#10b981", fontWeight: 600, fontSize: 12 }}>★ STRONG FIT</span>;
  if (score >= 50) return <span style={{ color: "#3b82f6", fontWeight: 600, fontSize: 12 }}>◑ GOOD FIT</span>;
  if (score >= 30) return <span style={{ color: "#f59e0b", fontWeight: 600, fontSize: 12 }}>△ PARTIAL FIT</span>;
  return <span style={{ color: "#ef4444", fontWeight: 600, fontSize: 12 }}>✗ WEAK FIT</span>;
}

export default function ResumeScreener() {
  const [jd, setJd] = useState(JD_DEFAULT);
  const [resumes, setResumes] = useState(RESUME_PRESETS.map((r) => ({ ...r, active: true })));
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("input");
  const [expandedIdx, setExpandedIdx] = useState(null);
  const [streamLog, setStreamLog] = useState("");

  const addResume = () =>
    setResumes((r) => [...r, { name: `Candidate ${r.length + 1}`, text: "", active: true }]);

  const updateResume = (i, field, val) =>
    setResumes((r) => r.map((item, idx) => (idx === i ? { ...item, [field]: val } : item)));

  const removeResume = (i) =>
    setResumes((r) => r.filter((_, idx) => idx !== i));

  const screenResumes = async () => {
    const activeResumes = resumes.filter((r) => r.active && r.text.trim());
    if (!activeResumes.length || !jd.trim()) return;

    setLoading(true);
    setStreamLog("Contacting AI engine...");
    setActiveTab("results");
    setResults(null);

    const systemPrompt = `You are an expert HR recruiter and NLP engineer. Analyze resumes against job descriptions.
Always respond with ONLY valid JSON — no markdown fences, no preamble.
Schema:
{
  "candidates": [
    {
      "name": "string",
      "composite_score": number (0-100),
      "tfidf_similarity": number (0-100),
      "tech_skill_overlap": number (0-100),
      "soft_skill_overlap": number (0-100),
      "matched_skills": ["skill1","skill2"],
      "missing_skills": ["skill1","skill2"],
      "extra_skills": ["skill1","skill2"],
      "summary": "2-3 sentence hiring recommendation",
      "strengths": ["point1","point2","point3"],
      "concerns": ["point1","point2"]
    }
  ],
  "jd_key_skills": ["skill1","skill2"],
  "top_recommendation": "Candidate Name",
  "analysis_note": "1-2 sentence overall analysis"
}`;

    const userPrompt = `JOB DESCRIPTION:
${jd}

RESUMES TO SCREEN:
${activeResumes.map((r, i) => `[RESUME ${i + 1} — ${r.name}]\n${r.text}`).join("\n\n---\n\n")}

Score each candidate on:
1. TF-IDF textual similarity to JD (0-100)
2. Tech skill overlap percentage (0-100)  
3. Soft skill overlap percentage (0-100)
4. Composite score = 0.55*tfidf + 0.35*tech_overlap + 0.10*soft_overlap (scaled 0-100)

Return the JSON schema exactly.`;

    try {
      setStreamLog("Analyzing resumes with NLP engine...");
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: systemPrompt,
          messages: [{ role: "user", content: userPrompt }],
        }),
      });
      const data = await response.json();
      const raw = data.content?.[0]?.text || "{}";
      const clean = raw.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);

      const ranked = (parsed.candidates || []).sort(
        (a, b) => b.composite_score - a.composite_score
      );
      setResults({ ...parsed, candidates: ranked });
      setStreamLog("");
    } catch (e) {
      setStreamLog("Error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const inputTab = activeTab === "input";
  const hasResults = results && results.candidates?.length > 0;

  return (
    <div style={{ background: "#0a0f1a", minHeight: "100vh", fontFamily: "'IBM Plex Mono', 'Courier New', monospace", color: "#e2e8f0" }}>

      {/* Header */}
      <div style={{ borderBottom: "1px solid #1e293b", padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 32, height: 32, background: ACCENT, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>⊕</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.05em", color: "#f1f5f9" }}>RESUMEIQ</div>
            <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.15em" }}>ML SCREENING SYSTEM v1.0</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["input", "results"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: "6px 14px",
                borderRadius: 6,
                border: `1px solid ${activeTab === tab ? ACCENT : "#1e293b"}`,
                background: activeTab === tab ? "#052e16" : "transparent",
                color: activeTab === tab ? ACCENT : "#64748b",
                fontSize: 11,
                letterSpacing: "0.1em",
                cursor: "pointer",
                fontFamily: "inherit",
                textTransform: "uppercase",
              }}
            >
              {tab === "input" ? "01 / Configure" : "02 / Results"}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: 24 }}>
        {/* INPUT TAB */}
        {activeTab === "input" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            {/* JD Panel */}
            <div>
              <div style={{ fontSize: 11, color: "#64748b", letterSpacing: "0.12em", marginBottom: 10 }}>JOB DESCRIPTION</div>
              <textarea
                value={jd}
                onChange={(e) => setJd(e.target.value)}
                style={{
                  width: "100%",
                  height: 340,
                  background: "#0d1626",
                  border: "1px solid #1e293b",
                  borderRadius: 8,
                  color: "#e2e8f0",
                  padding: 14,
                  fontSize: 12,
                  lineHeight: 1.7,
                  fontFamily: "inherit",
                  resize: "none",
                  boxSizing: "border-box",
                  outline: "none",
                }}
                placeholder="Paste job description here..."
              />
            </div>

            {/* Resumes Panel */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontSize: 11, color: "#64748b", letterSpacing: "0.12em" }}>
                  RESUMES ({resumes.filter((r) => r.active).length} active)
                </div>
                <button
                  onClick={addResume}
                  style={{ padding: "4px 10px", border: `1px solid ${ACCENT}`, borderRadius: 4, background: "transparent", color: ACCENT, fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}
                >
                  + ADD
                </button>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 10, maxHeight: 360, overflowY: "auto" }}>
                {resumes.map((r, i) => (
                  <div key={i} style={{ background: "#0d1626", border: `1px solid ${r.active ? "#1e3a5f" : "#1e293b"}`, borderRadius: 8, padding: 12, opacity: r.active ? 1 : 0.45 }}>
                    <div style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={r.active}
                        onChange={(e) => updateResume(i, "active", e.target.checked)}
                        style={{ accentColor: ACCENT }}
                      />
                      <input
                        value={r.name}
                        onChange={(e) => updateResume(i, "name", e.target.value)}
                        style={{ flex: 1, background: "transparent", border: "none", color: "#94a3b8", fontSize: 12, fontFamily: "inherit", outline: "none", fontWeight: 600 }}
                      />
                      <button onClick={() => removeResume(i)} style={{ background: "transparent", border: "none", color: "#ef4444", cursor: "pointer", fontSize: 14, padding: 0 }}>×</button>
                    </div>
                    <textarea
                      value={r.text}
                      onChange={(e) => updateResume(i, "text", e.target.value)}
                      style={{ width: "100%", height: 72, background: "#060d1a", border: "1px solid #1e293b", borderRadius: 4, color: "#94a3b8", padding: "8px 10px", fontSize: 11, fontFamily: "inherit", resize: "none", boxSizing: "border-box", outline: "none", lineHeight: 1.6 }}
                      placeholder="Paste resume text..."
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Run Button */}
            <div style={{ gridColumn: "1 / -1", display: "flex", justifyContent: "center", marginTop: 8 }}>
              <button
                onClick={screenResumes}
                disabled={loading}
                style={{
                  padding: "12px 40px",
                  background: loading ? "#1e293b" : ACCENT,
                  color: loading ? "#64748b" : "#022c22",
                  border: "none",
                  borderRadius: 8,
                  fontSize: 13,
                  fontWeight: 700,
                  letterSpacing: "0.1em",
                  cursor: loading ? "not-allowed" : "pointer",
                  fontFamily: "inherit",
                  textTransform: "uppercase",
                  transition: "all 0.2s",
                }}
              >
                {loading ? "⟳ SCREENING..." : "▶ SCREEN RESUMES"}
              </button>
            </div>
          </div>
        )}

        {/* RESULTS TAB */}
        {activeTab === "results" && (
          <div>
            {loading && (
              <div style={{ textAlign: "center", padding: 60, color: "#64748b" }}>
                <div style={{ fontSize: 28, marginBottom: 12, animation: "spin 1s linear infinite" }}>◌</div>
                <div style={{ fontSize: 12, letterSpacing: "0.1em" }}>{streamLog}</div>
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              </div>
            )}

            {!loading && !hasResults && (
              <div style={{ textAlign: "center", padding: 60, color: "#1e293b" }}>
                <div style={{ fontSize: 40, marginBottom: 16 }}>◯</div>
                <div style={{ fontSize: 12, letterSpacing: "0.1em", color: "#475569" }}>NO RESULTS YET — RUN SCREENING FIRST</div>
              </div>
            )}

            {!loading && hasResults && (
              <div>
                {/* Summary bar */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
                  {[
                    { label: "SCREENED", value: results.candidates.length },
                    { label: "STRONG FIT", value: results.candidates.filter((c) => c.composite_score >= 70).length, color: "#10b981" },
                    { label: "TOP SCORE", value: `${Math.round(results.candidates[0].composite_score)}%`, color: ACCENT },
                    { label: "JD SKILLS", value: results.jd_key_skills?.length || "—" },
                  ].map((s, i) => (
                    <div key={i} style={{ background: "#0d1626", border: "1px solid #1e293b", borderRadius: 8, padding: "14px 16px" }}>
                      <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", marginBottom: 4 }}>{s.label}</div>
                      <div style={{ fontSize: 24, fontWeight: 700, color: s.color || "#e2e8f0" }}>{s.value}</div>
                    </div>
                  ))}
                </div>

                {/* JD skills */}
                {results.jd_key_skills?.length > 0 && (
                  <div style={{ background: "#0d1626", border: "1px solid #1e293b", borderRadius: 8, padding: "12px 16px", marginBottom: 20 }}>
                    <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.12em", marginBottom: 8 }}>JD REQUIRED SKILLS</div>
                    <div>{results.jd_key_skills.map((s) => <Chip key={s} label={s} type="skill" />)}</div>
                  </div>
                )}

                {/* Candidate cards */}
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {results.candidates.map((c, i) => {
                    const isExpanded = expandedIdx === i;
                    return (
                      <div
                        key={i}
                        style={{ background: "#0d1626", border: `1px solid ${i === 0 ? ACCENT : "#1e293b"}`, borderRadius: 10, overflow: "hidden", transition: "border-color 0.2s" }}
                      >
                        {/* Card header */}
                        <div
                          style={{ padding: "14px 18px", cursor: "pointer", display: "grid", gridTemplateColumns: "28px 1fr 140px 100px", alignItems: "center", gap: 16 }}
                          onClick={() => setExpandedIdx(isExpanded ? null : i)}
                        >
                          <div style={{ fontSize: 13, fontWeight: 700, color: i === 0 ? ACCENT : "#475569" }}>#{i + 1}</div>
                          <div>
                            <div style={{ fontSize: 14, fontWeight: 600, color: "#f1f5f9", marginBottom: 2 }}>{c.name}</div>
                            <TierBadge score={c.composite_score} />
                          </div>
                          <div>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                              <span style={{ fontSize: 10, color: "#64748b" }}>MATCH</span>
                              <span style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0" }}>{Math.round(c.composite_score)}%</span>
                            </div>
                            <ScoreBar value={c.composite_score} />
                          </div>
                          <div style={{ textAlign: "right", fontSize: 13, color: "#475569" }}>{isExpanded ? "▲" : "▼"}</div>
                        </div>

                        {/* Expanded details */}
                        {isExpanded && (
                          <div style={{ borderTop: "1px solid #1e293b", padding: "16px 18px" }}>
                            {/* Score breakdown */}
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 16 }}>
                              {[
                                { label: "TF-IDF SIM", val: c.tfidf_similarity },
                                { label: "TECH OVERLAP", val: c.tech_skill_overlap },
                                { label: "SOFT OVERLAP", val: c.soft_skill_overlap },
                              ].map((m) => (
                                <div key={m.label} style={{ background: "#060d1a", borderRadius: 6, padding: "10px 12px" }}>
                                  <div style={{ fontSize: 9, color: "#475569", letterSpacing: "0.12em", marginBottom: 6 }}>{m.label}</div>
                                  <div style={{ fontSize: 18, fontWeight: 700, color: "#94a3b8", marginBottom: 6 }}>{Math.round(m.val)}%</div>
                                  <ScoreBar value={m.val} />
                                </div>
                              ))}
                            </div>

                            {/* Skills */}
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
                              <div>
                                <div style={{ fontSize: 10, color: "#10b981", letterSpacing: "0.1em", marginBottom: 6 }}>✓ MATCHED SKILLS</div>
                                <div>{c.matched_skills?.length ? c.matched_skills.map((s) => <Chip key={s} label={s} type="matched" />) : <span style={{ fontSize: 11, color: "#475569" }}>None</span>}</div>
                              </div>
                              <div>
                                <div style={{ fontSize: 10, color: "#ef4444", letterSpacing: "0.1em", marginBottom: 6 }}>✗ MISSING SKILLS</div>
                                <div>{c.missing_skills?.length ? c.missing_skills.map((s) => <Chip key={s} label={s} type="missing" />) : <span style={{ fontSize: 11, color: "#10b981" }}>All covered ✓</span>}</div>
                              </div>
                            </div>

                            {/* AI Summary */}
                            {c.summary && (
                              <div style={{ background: "#060d1a", borderRadius: 6, padding: "10px 14px", marginBottom: 10, borderLeft: `3px solid ${ACCENT}` }}>
                                <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.1em", marginBottom: 4 }}>AI ASSESSMENT</div>
                                <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.7 }}>{c.summary}</div>
                              </div>
                            )}

                            {/* Strengths & Concerns */}
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                              {c.strengths?.length > 0 && (
                                <div>
                                  <div style={{ fontSize: 10, color: "#3b82f6", letterSpacing: "0.1em", marginBottom: 6 }}>STRENGTHS</div>
                                  {c.strengths.map((s, si) => <div key={si} style={{ fontSize: 11, color: "#64748b", marginBottom: 3 }}>→ {s}</div>)}
                                </div>
                              )}
                              {c.concerns?.length > 0 && (
                                <div>
                                  <div style={{ fontSize: 10, color: "#f59e0b", letterSpacing: "0.1em", marginBottom: 6 }}>CONCERNS</div>
                                  {c.concerns.map((s, si) => <div key={si} style={{ fontSize: 11, color: "#64748b", marginBottom: 3 }}>→ {s}</div>)}
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Overall note */}
                {results.analysis_note && (
                  <div style={{ marginTop: 20, background: "#0d1626", border: "1px solid #1e293b", borderRadius: 8, padding: "12px 16px" }}>
                    <div style={{ fontSize: 10, color: "#64748b", letterSpacing: "0.12em", marginBottom: 6 }}>ANALYST NOTE</div>
                    <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.7 }}>{results.analysis_note}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
