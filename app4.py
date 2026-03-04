import { useState, useEffect, useRef, useCallback } from "react";

// ── Simulated backend API ──────────────────────────────────────────────────
const BACKEND = {
  stats: {
    totalScanned: 4582,
    deepfakesDetected: 327,
    activeAnalyzing: 12,
    detectionAccuracy: 98,
  },
  alerts: [
    { id: 1, type: "danger",  icon: "⚠", text: "High Fake Probability Detected",  time: "2m ago" },
    { id: 2, type: "warning", icon: "⚠", text: "Video Tampering Identified",       time: "5m ago" },
    { id: 3, type: "warning", icon: "⚠", text: "Suspicious Audio Alteration",      time: "12m ago" },
    { id: 4, type: "success", icon: "✓", text: "Image Analysis Completed",         time: "18m ago" },
    { id: 5, type: "danger",  icon: "⚠", text: "Face Swap Detected",               time: "31m ago" },
    { id: 6, type: "success", icon: "✓", text: "Batch Scan Finished",              time: "1h ago" },
  ],
  uploads: [
    { id: 1, name: "Interview_clip.mp4",      type: "video",  result: "Deepfake",     date: "Today",       badge: "fake" },
    { id: 2, name: "Suspicious_image.jpg",    type: "image",  result: "Authentic",    date: "Today",       badge: "real" },
    { id: 3, name: "Fake_voice_audio.mp3",    type: "audio",  result: "Fake",         date: "Yesterday",   badge: "fake" },
    { id: 4, name: "Security_footage.mov",    type: "video",  result: "Under Review", date: "Apr 15, 2024",badge: "review" },
    { id: 5, name: "Profile_photo.png",       type: "image",  result: "Authentic",    date: "Apr 14, 2024",badge: "real" },
    { id: 6, name: "Press_conference.mp4",    type: "video",  result: "Deepfake",     date: "Apr 13, 2024",badge: "fake" },
  ],
  weekStats: [
    { day: "Mon", real: 38, fake: 22 },
    { day: "Tue", real: 45, fake: 31 },
    { day: "Wed", real: 52, fake: 28 },
    { day: "Thu", real: 41, fake: 35 },
    { day: "Fri", real: 60, fake: 42 },
    { day: "Sat", real: 35, fake: 18 },
    { day: "Sun", real: 29, fake: 24 },
  ],
  logs: [
    { id: "LOG-4891", file: "interview_clip.mp4",   result: "FAKE",   conf: 96.2, ts: "2024-04-15 14:32" },
    { id: "LOG-4890", file: "profile_photo.png",    result: "REAL",   conf: 91.5, ts: "2024-04-15 14:28" },
    { id: "LOG-4889", file: "press_conf.mp4",       result: "FAKE",   conf: 88.3, ts: "2024-04-15 13:55" },
    { id: "LOG-4888", file: "id_document.jpg",      result: "REAL",   conf: 97.1, ts: "2024-04-15 13:40" },
    { id: "LOG-4887", file: "audio_sample.mp3",     result: "FAKE",   conf: 79.4, ts: "2024-04-15 12:15" },
    { id: "LOG-4886", file: "selfie_vid.mov",       result: "REVIEW", conf: 54.8, ts: "2024-04-15 11:00" },
    { id: "LOG-4885", file: "news_clip.mp4",        result: "FAKE",   conf: 93.7, ts: "2024-04-14 22:30" },
    { id: "LOG-4884", file: "document_scan.pdf",    result: "REAL",   conf: 99.0, ts: "2024-04-14 21:10" },
  ],

  // Simulated analysis
  analyzeImage(file) {
    return new Promise((resolve) => {
      setTimeout(() => {
        const fakePct   = Math.floor(Math.random() * 60) + 40;
        const realPct   = 100 - fakePct;
        const isFake    = fakePct > 60;
        resolve({
          isFake,
          fakePct,
          realPct,
          confidence: fakePct,
          faceManipulation: Math.floor(Math.random() * 15) + 83,
          deepfakeArtifacts: Math.floor(Math.random() * 15) + 80,
          aiGeneratedIndicators: Math.floor(Math.random() * 15) + 82,
          filename: file.name,
        });
      }, 2200);
    });
  },
};

// ── Tiny SVG icons ─────────────────────────────────────────────────────────
const Icon = {
  Dashboard:    () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>,
  Upload:       () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="9" x2="9" y2="21"/></svg>,
  Analysis:     () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>,
  Logs:         () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>,
  Settings:     () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  Bell:         () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>,
  Report:       () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="18" height="18"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/></svg>,
  Video:        () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14"><rect x="2" y="7" width="15" height="10" rx="2"/><polygon points="17 9 22 5 22 19 17 15"/></svg>,
  Image:        () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>,
  Audio:        () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>,
  ChevronRight: () => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14"><polyline points="9 18 15 12 9 6"/></svg>,
};

// ── Spark line mini chart ──────────────────────────────────────────────────
const SparkLine = ({ color, points }) => {
  const max = Math.max(...points), min = Math.min(...points);
  const norm = (v) => 100 - ((v - min) / (max - min + 1)) * 90;
  const W = 80, H = 30;
  const pts = points.map((p, i) => `${(i / (points.length - 1)) * W},${(norm(p) / 100) * H}`).join(" ");
  return (
    <svg width={W} height={H} style={{ opacity: 0.7 }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
};

// ── Detection Stats Mini Chart ─────────────────────────────────────────────
const StatsChart = ({ data }) => {
  const maxVal = Math.max(...data.flatMap((d) => [d.real, d.fake]));
  const W = 560, H = 120, PAD = 30;
  const xStep = (W - PAD * 2) / (data.length - 1);
  const yScale = (v) => H - PAD - (v / maxVal) * (H - PAD * 1.5);

  const realPts = data.map((d, i) => `${PAD + i * xStep},${yScale(d.real)}`).join(" ");
  const fakePts = data.map((d, i) => `${PAD + i * xStep},${yScale(d.fake)}`).join(" ");

  const realArea = `${PAD},${H - PAD} ` + realPts + ` ${PAD + (data.length - 1) * xStep},${H - PAD}`;
  const fakeArea = `${PAD},${H - PAD} ` + fakePts + ` ${PAD + (data.length - 1) * xStep},${H - PAD}`;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ display: "block" }}>
      <defs>
        <linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="fg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f04b5a" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#f04b5a" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={realArea} fill="url(#rg)" />
      <polygon points={fakeArea} fill="url(#fg)" />
      <polyline points={realPts} fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinejoin="round" />
      <polyline points={fakePts} fill="none" stroke="#f04b5a" strokeWidth="2" strokeLinejoin="round" />
      {data.map((d, i) => (
        <g key={i}>
          <circle cx={PAD + i * xStep} cy={yScale(d.real)} r="3" fill="#38bdf8" />
          <circle cx={PAD + i * xStep} cy={yScale(d.fake)} r="3" fill="#f04b5a" />
          <text x={PAD + i * xStep} y={H - 4} textAnchor="middle" fill="#4a5568" fontSize="10" fontFamily="'DM Mono',monospace">
            {d.day}
          </text>
        </g>
      ))}
    </svg>
  );
};

// ── Grad-CAM canvas simulation ─────────────────────────────────────────────
const GradCAMCanvas = ({ active }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!active || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0e1a";
    ctx.fillRect(0, 0, W, H);

    // Heat blobs
    const blobs = [
      { x: W * 0.5, y: H * 0.35, r: 90, c: [255, 30, 30] },
      { x: W * 0.45, y: H * 0.42, r: 70, c: [255, 120, 0] },
      { x: W * 0.55, y: H * 0.45, r: 55, c: [255, 200, 0] },
      { x: W * 0.38, y: H * 0.55, r: 45, c: [200, 255, 0] },
      { x: W * 0.62, y: H * 0.38, r: 40, c: [0, 200, 255] },
      { x: W * 0.3,  y: H * 0.3,  r: 30, c: [0, 100, 255] },
      { x: W * 0.7,  y: H * 0.6,  r: 25, c: [0, 50, 200] },
    ];
    blobs.forEach(({ x, y, r, c }) => {
      const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, `rgba(${c[0]},${c[1]},${c[2]},0.85)`);
      grad.addColorStop(0.6, `rgba(${c[0]},${c[1]},${c[2]},0.35)`);
      grad.addColorStop(1, `rgba(${c[0]},${c[1]},${c[2]},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    });

    // Label
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(0, H - 40, W, 40);
    ctx.fillStyle = "#e2e8f0";
    ctx.font      = "bold 13px 'DM Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("GRAD-CAM", W / 2, H - 16);
  }, [active]);
  return (
    <canvas
      ref={canvasRef}
      width={300}
      height={340}
      style={{ width: "100%", height: "100%", display: "block", borderRadius: "4px" }}
    />
  );
};

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [activeNav,    setActiveNav]    = useState("Dashboard");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analyzing,    setAnalyzing]    = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedURL,  setUploadedURL]  = useState(null);
  const [dragOver,     setDragOver]     = useState(false);
  const [alertCount,   setAlertCount]   = useState(3);
  const [weekFilter,   setWeekFilter]   = useState("This Week");
  const [statsData,    setStatsData]    = useState(BACKEND.stats);
  const [logs,         setLogs]         = useState(BACKEND.logs);
  const [settingsSaved,setSettingsSaved]= useState(false);
  const fileInputRef = useRef(null);

  // Live counter animation
  const [liveCount, setLiveCount] = useState(statsData.activeAnalyzing);
  useEffect(() => {
    const t = setInterval(() => {
      setLiveCount((n) => {
        const delta = Math.random() > 0.5 ? 1 : -1;
        return Math.max(8, Math.min(18, n + delta));
      });
    }, 3000);
    return () => clearInterval(t);
  }, []);

  const handleFile = useCallback(async (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    setUploadedFile(file);
    setUploadedURL(url);
    setAnalysisResult(null);
    setAnalyzing(true);

    const result = await BACKEND.analyzeImage(file);
    setAnalysisResult(result);
    setAnalyzing(false);
    setAlertCount((n) => n + 1);

    // Add to logs
    const newLog = {
      id:     `LOG-${4892 + logs.length}`,
      file:   file.name,
      result: result.isFake ? "FAKE" : "REAL",
      conf:   result.confidence,
      ts:     new Date().toLocaleString(),
    };
    setLogs((prev) => [newLog, ...prev]);
    setStatsData((prev) => ({
      ...prev,
      totalScanned:       prev.totalScanned + 1,
      deepfakesDetected:  result.isFake ? prev.deepfakesDetected + 1 : prev.deepfakesDetected,
    }));
  }, [logs.length]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const navItems = [
    { label: "Dashboard",     icon: Icon.Dashboard },
    { label: "Uploads",       icon: Icon.Upload    },
    { label: "Live Analysis", icon: Icon.Analysis  },
    { label: "Detection Logs",icon: Icon.Logs      },
    { label: "System Settings",icon: Icon.Settings },
  ];

  const typeIcon = (t) => {
    if (t === "video") return <Icon.Video />;
    if (t === "audio") return <Icon.Audio />;
    return <Icon.Image />;
  };

  return (
    <div style={S.root}>
      {/* ── Sidebar ── */}
      <aside style={S.sidebar}>
        <div style={S.logo}>
          <div style={S.logoMark}>
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#38bdf8" strokeWidth="2">
              <path d="M12 2L3 7l9 5 9-5-9-5z"/>
              <path d="M3 12l9 5 9-5"/>
              <path d="M3 17l9 5 9-5"/>
            </svg>
          </div>
          <div>
            <div style={S.logoText}>AI AUTHENTICA</div>
          </div>
        </div>
        <nav style={S.nav}>
          {navItems.map(({ label, icon: NavIcon }) => (
            <button
              key={label}
              onClick={() => setActiveNav(label)}
              style={{ ...S.navItem, ...(activeNav === label ? S.navItemActive : {}) }}
            >
              <span style={{ color: activeNav === label ? "#38bdf8" : "#64748b" }}>
                <NavIcon />
              </span>
              <span style={{ fontSize: "13px", fontWeight: activeNav === label ? "600" : "400" }}>
                {label}
              </span>
            </button>
          ))}
        </nav>
        <div style={S.sidebarFooter}>
          <div style={S.pulseRing} />
          <span style={{ fontSize: "11px", color: "#38bdf8", fontFamily: "'DM Mono',monospace", letterSpacing: "1px" }}>
            SYSTEM ACTIVE
          </span>
        </div>
      </aside>

      {/* ── Main ── */}
      <div style={S.main}>
        {/* Top Bar */}
        <header style={S.topbar}>
          <h1 style={S.pageTitle}>DEEPFAKE DETECTION DASHBOARD</h1>
          <div style={S.topbarRight}>
            <button style={S.topbarBtn} onClick={() => setAlertCount(0)}>
              <Icon.Bell />
              <span style={{ marginLeft: 6 }}>Alerts</span>
              {alertCount > 0 && <span style={S.badge}>{alertCount}</span>}
            </button>
            <button style={S.topbarBtn}>
              <Icon.Report />
              <span style={{ marginLeft: 6 }}>Reports</span>
            </button>
            <button style={S.topbarBtn} onClick={() => setActiveNav("System Settings")}>
              <Icon.Settings />
              <span style={{ marginLeft: 6 }}>Settings</span>
            </button>
            <div style={S.adminChip}>
              <div style={S.adminAvatar}>A</div>
              <span style={{ fontSize: "13px", fontWeight: 600 }}>Admin</span>
            </div>
          </div>
        </header>

        {/* ── DASHBOARD ── */}
        {activeNav === "Dashboard" && (
          <div style={S.content}>
            {/* Stat Cards */}
            <div style={S.statGrid}>
              {[
                {
                  label: "Total Files Scanned",
                  value: statsData.totalScanned.toLocaleString(),
                  color: "#38bdf8",
                  sparkData: [30, 42, 35, 50, 45, 60, 55, 70],
                },
                {
                  label: "Deepfakes Detected",
                  value: statsData.deepfakesDetected,
                  color: "#f04b5a",
                  badge: "High Risk",
                  sparkData: [10, 18, 14, 25, 20, 28, 22, 30],
                },
                {
                  label: "Active Analyzing",
                  value: liveCount,
                  color: "#f0a84b",
                  badge: "In Progress",
                  sparkData: [5, 8, 12, 9, 15, 10, 13, liveCount],
                },
                {
                  label: "Detection Accuracy",
                  value: `${statsData.detectionAccuracy}%`,
                  subLabel: "Precision Rate",
                  color: "#2dd4a0",
                  sparkData: [92, 94, 93, 96, 95, 97, 96, 98],
                },
              ].map((s, i) => (
                <div key={i} style={S.statCard}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div>
                      <div style={S.statLabel}>{s.label}</div>
                      <div style={{ ...S.statValue, color: s.color }}>{s.value}</div>
                      {s.badge && (
                        <span style={{
                          ...S.statBadge,
                          background: s.color === "#f04b5a" ? "rgba(240,75,90,0.15)" : "rgba(240,168,75,0.15)",
                          color: s.color,
                          border: `1px solid ${s.color}40`,
                        }}>
                          {s.badge}
                        </span>
                      )}
                      {s.subLabel && <div style={{ fontSize: "11px", color: "#64748b", marginTop: 4 }}>{s.subLabel}</div>}
                    </div>
                    <SparkLine color={s.color} points={s.sparkData} />
                  </div>
                </div>
              ))}
            </div>

            {/* Main Analysis + Sidebar */}
            <div style={S.mainRow}>
              {/* Deepfake Analysis Panel */}
              <div style={S.analysisPanel}>
                <div style={S.panelHeader}>
                  <span style={S.panelTitle}>Deepfake Analysis</span>
                  <div style={{ display: "flex", gap: 6 }}>
                    {["▲", "•", "•", "•"].map((d, i) => (
                      <span key={i} style={{ color: "#4a5568", fontSize: 12, cursor: "pointer" }}>{d}</span>
                    ))}
                  </div>
                </div>

                {/* Image panels */}
                <div style={S.imagePair}>
                  {/* Upload area */}
                  <div
                    style={{
                      ...S.imageBox,
                      borderColor: dragOver ? "#38bdf8" : analysisResult?.isFake ? "#f04b5a" : "#1e293b",
                      cursor: "pointer",
                    }}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <div style={S.imageBoxLabel}>UPLOADED IMAGE</div>
                    {uploadedURL ? (
                      <>
                        <img src={uploadedURL} alt="uploaded"
                          style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
                        {analyzing && (
                          <div style={S.analyzingOverlay}>
                            <div style={S.spinner} />
                            <span style={{ color: "#38bdf8", fontFamily: "'DM Mono',monospace", fontSize: 12, marginTop: 8 }}>
                              ANALYZING…
                            </span>
                          </div>
                        )}
                        {analysisResult && (
                          <div style={{
                            ...S.verdictOverlay,
                            background: analysisResult.isFake ? "rgba(240,75,90,0.85)" : "rgba(45,212,160,0.85)"
                          }}>
                            {analysisResult.isFake ? "FAKE" : "REAL"}
                          </div>
                        )}
                      </>
                    ) : (
                      <div style={S.uploadPlaceholder}>
                        <div style={{ fontSize: 36, opacity: 0.3, marginBottom: 8 }}>⬆</div>
                        <div style={{ fontSize: 11, color: "#4a5568", fontFamily: "'DM Mono',monospace", textAlign: "center" }}>
                          DROP IMAGE<br />OR CLICK TO UPLOAD
                        </div>
                      </div>
                    )}
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      style={{ display: "none" }}
                      onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
                    />
                  </div>

                  {/* Grad-CAM */}
                  <div style={{ ...S.imageBox, borderColor: "#1e293b", cursor: "default" }}>
                    <div style={S.imageBoxLabel}>GRAD-CAM</div>
                    {analysisResult ? (
                      <GradCAMCanvas active={!!analysisResult} />
                    ) : (
                      <div style={S.uploadPlaceholder}>
                        <div style={{ fontSize: 11, color: "#4a5568", fontFamily: "'DM Mono',monospace", textAlign: "center", lineHeight: 1.8 }}>
                          EXPLAINABILITY<br />HEATMAP<br />AWAITING ANALYSIS
                        </div>
                      </div>
                    )}
                    {analysisResult && (
                      <div style={{ ...S.verdictOverlay, background: "rgba(167,139,250,0.85)" }}>
                        GRAD-CAM
                      </div>
                    )}
                  </div>
                </div>

                {/* Probability bar */}
                {analysisResult ? (
                  <>
                    <div style={S.probBar}>
                      <div style={{ width: `${analysisResult.fakePct}%`, background: "#f04b5a", height: "100%", borderRadius: "3px 0 0 3px", transition: "width 0.8s ease" }} />
                      <div style={{ width: `${analysisResult.realPct}%`, background: "#1e293b", height: "100%", borderRadius: "0 3px 3px 0" }} />
                    </div>
                    <div style={S.probLabels}>
                      <span style={{ color: "#f04b5a", fontWeight: 700 }}>
                        <span style={S.probDot("#f04b5a")} /> {analysisResult.fakePct}% Fake
                      </span>
                      <span style={{ color: "#94a3b8" }}>
                        <span style={S.probDot("#94a3b8")} /> {analysisResult.realPct}% Real
                      </span>
                    </div>

                    {/* Detailed results */}
                    <div style={S.detailedBox}>
                      <div style={S.detailedTitle}>Detailed Results</div>
                      {[
                        { label: "Face Manipulation",       val: analysisResult.faceManipulation,       color: "#f04b5a" },
                        { label: "Deepfake Artifacts",      val: analysisResult.deepfakeArtifacts,      color: "#2dd4a0" },
                        { label: "AI Generated Indicators", val: analysisResult.aiGeneratedIndicators,  color: "#2dd4a0" },
                      ].map((item) => (
                        <div key={item.label} style={{ marginBottom: 8 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                            <span style={{ color: item.color, fontSize: 14 }}>
                              {item.color === "#f04b5a" ? "✗" : "✓"}
                            </span>
                            <span style={{ fontSize: 13, color: "#94a3b8", flex: 1 }}>{item.label}:</span>
                            <span style={{ fontSize: 13, color: item.color, fontFamily: "'DM Mono',monospace", fontWeight: 600 }}>
                              {item.val}%
                            </span>
                          </div>
                          <div style={{ height: 3, background: "#0f172a", borderRadius: 2 }}>
                            <div style={{ width: `${item.val}%`, height: "100%", background: item.color, borderRadius: 2, transition: "width 1s ease" }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div style={{ padding: "12px 0", textAlign: "center", color: "#4a5568", fontSize: "12px", fontFamily: "'DM Mono',monospace" }}>
                    Upload an image to run deepfake analysis
                  </div>
                )}

                {/* Detection stats chart */}
                <div style={{ marginTop: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <span style={S.panelTitle}>Detection Statistics</span>
                    <select
                      value={weekFilter}
                      onChange={(e) => setWeekFilter(e.target.value)}
                      style={S.select}
                    >
                      {["This Week", "Last Week", "Last Month"].map((o) => (
                        <option key={o}>{o}</option>
                      ))}
                    </select>
                  </div>
                  <StatsChart data={BACKEND.weekStats} />
                  <div style={{ display: "flex", gap: 24, marginTop: 8 }}>
                    {[
                      { color: "#38bdf8", label: "Real Files",   val: "245" },
                      { color: "#f04b5a", label: "Deepfakes",    val: "198" },
                      { color: "#f0a84b", label: "Under Review", val: "32"  },
                    ].map((s) => (
                      <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ color: s.color, fontSize: 12 }}>▶</span>
                        <span style={{ color: s.color, fontFamily: "'DM Mono',monospace", fontSize: 12, fontWeight: 700 }}>
                          {s.val}
                        </span>
                        <span style={{ color: "#64748b", fontSize: 12 }}>{s.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right sidebar */}
              <div style={S.rightCol}>
                {/* Recent Alerts */}
                <div style={S.sidePanel}>
                  <div style={S.panelHeader}>
                    <span style={S.panelTitle}>Recent Alerts</span>
                    <button style={S.viewAllBtn} onClick={() => setActiveNav("Detection Logs")}>
                      View All
                    </button>
                  </div>
                  <div>
                    {BACKEND.alerts.map((a) => (
                      <div key={a.id} style={{
                        ...S.alertRow,
                        borderLeft: `3px solid ${a.type === "danger" ? "#f04b5a" : a.type === "warning" ? "#f0a84b" : "#2dd4a0"}`,
                      }}>
                        <span style={{ fontSize: 14, color: a.type === "danger" ? "#f04b5a" : a.type === "warning" ? "#f0a84b" : "#2dd4a0" }}>
                          {a.type === "success" ? "✓" : "⚠"}
                        </span>
                        <span style={{ flex: 1, fontSize: 12, color: "#94a3b8" }}>{a.text}</span>
                        <span style={{ ...S.alertTime }}>{a.time}</span>
                        <Icon.ChevronRight />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recent Uploads */}
                <div style={{ ...S.sidePanel, marginTop: 14 }}>
                  <div style={S.panelHeader}>
                    <span style={S.panelTitle}>Recent Uploads</span>
                    <div style={{ display: "flex", gap: 4 }}>
                      {["•", "•", "•"].map((d, i) => (
                        <span key={i} style={{ color: "#38bdf8", fontSize: 10 }}>{d}</span>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto auto", gap: "6px 12px", alignItems: "center" }}>
                    {["File Name", "Type", "Result", "Date"].map((h) => (
                      <span key={h} style={{ fontSize: 11, color: "#4a5568", fontFamily: "'DM Mono',monospace", paddingBottom: 6, borderBottom: "1px solid #1e293b" }}>
                        {h}
                      </span>
                    ))}
                    {BACKEND.uploads.map((u) => (
                      <>
                        <div key={u.id + "n"} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#94a3b8", overflow: "hidden" }}>
                          <span style={{ color: "#38bdf8" }}>{typeIcon(u.type)}</span>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 120 }}>{u.name}</span>
                        </div>
                        <span key={u.id + "t"} style={{ fontSize: 11, color: "#64748b", fontFamily: "'DM Mono',monospace" }}>
                          {u.type.charAt(0).toUpperCase() + u.type.slice(1)}
                        </span>
                        <span key={u.id + "r"} style={{
                          fontSize: 11, fontWeight: 700, fontFamily: "'DM Mono',monospace",
                          color: u.badge === "fake" ? "#f04b5a" : u.badge === "real" ? "#2dd4a0" : "#f0a84b",
                          background: u.badge === "fake" ? "rgba(240,75,90,0.1)" : u.badge === "real" ? "rgba(45,212,160,0.1)" : "rgba(240,168,75,0.1)",
                          padding: "2px 7px", borderRadius: 4,
                        }}>
                          {u.result}
                        </span>
                        <span key={u.id + "d"} style={{ fontSize: 11, color: "#4a5568" }}>{u.date}</span>
                      </>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── UPLOADS PAGE ── */}
        {activeNav === "Uploads" && (
          <div style={S.content}>
            <div style={S.pageCard}>
              <div style={S.panelHeader}>
                <span style={S.panelTitle}>Upload Files for Analysis</span>
              </div>
              <div
                style={{
                  ...S.dropZone,
                  borderColor: dragOver ? "#38bdf8" : "#1e293b",
                  background: dragOver ? "rgba(56,189,248,0.04)" : "#080c14",
                }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { handleDrop(e); setActiveNav("Dashboard"); }}
                onClick={() => { fileInputRef.current?.click(); }}
              >
                <div style={{ fontSize: 48, opacity: 0.3, marginBottom: 12 }}>⬆</div>
                <div style={{ color: "#38bdf8", fontFamily: "'DM Mono',monospace", fontSize: 14, marginBottom: 8 }}>
                  DROP FILES HERE OR CLICK TO BROWSE
                </div>
                <div style={{ color: "#4a5568", fontSize: 12 }}>
                  Supports: JPG, PNG, MP4, MOV, MP3, WAV · Max size: 500MB
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,video/*,audio/*"
                  style={{ display: "none" }}
                  onChange={(e) => { if (e.target.files[0]) { handleFile(e.target.files[0]); setActiveNav("Dashboard"); } }}
                />
              </div>
              <div style={{ marginTop: 24 }}>
                <div style={{ ...S.panelTitle, marginBottom: 12 }}>Recent Uploads</div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      {["File Name", "Type", "Size", "Result", "Date", "Action"].map((h) => (
                        <th key={h} style={S.th}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {BACKEND.uploads.map((u) => (
                      <tr key={u.id} style={{ borderBottom: "1px solid #0f172a" }}>
                        <td style={S.td}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ color: "#38bdf8" }}>{typeIcon(u.type)}</span>
                            {u.name}
                          </div>
                        </td>
                        <td style={S.td}>{u.type}</td>
                        <td style={S.td}>{Math.floor(Math.random() * 50 + 1)} MB</td>
                        <td style={S.td}>
                          <span style={{
                            fontSize: 11, fontWeight: 700,
                            color: u.badge === "fake" ? "#f04b5a" : u.badge === "real" ? "#2dd4a0" : "#f0a84b",
                            background: u.badge === "fake" ? "rgba(240,75,90,0.1)" : u.badge === "real" ? "rgba(45,212,160,0.1)" : "rgba(240,168,75,0.1)",
                            padding: "2px 8px", borderRadius: 4, fontFamily: "'DM Mono',monospace",
                          }}>{u.result}</span>
                        </td>
                        <td style={S.td}>{u.date}</td>
                        <td style={S.td}>
                          <button style={{ ...S.viewAllBtn, fontSize: 11 }}
                            onClick={() => setActiveNav("Dashboard")}>
                            Re-Analyze
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── LIVE ANALYSIS ── */}
        {activeNav === "Live Analysis" && (
          <div style={S.content}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div style={S.pageCard}>
                <div style={S.panelHeader}>
                  <span style={S.panelTitle}>Live Upload & Analyze</span>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={S.pulseRing} />
                    <span style={{ fontSize: 11, color: "#2dd4a0", fontFamily: "'DM Mono',monospace" }}>LIVE</span>
                  </div>
                </div>
                <div
                  style={{ ...S.dropZone, minHeight: 180, cursor: "pointer" }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  {uploadedURL ? (
                    <img src={uploadedURL} alt="uploaded"
                      style={{ maxHeight: 200, maxWidth: "100%", objectFit: "contain", borderRadius: 6 }} />
                  ) : (
                    <>
                      <div style={{ fontSize: 36, opacity: 0.2, marginBottom: 8 }}>🎯</div>
                      <div style={{ color: "#4a5568", fontFamily: "'DM Mono',monospace", fontSize: 12 }}>
                        CLICK TO UPLOAD IMAGE FOR LIVE ANALYSIS
                      </div>
                    </>
                  )}
                  <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }}
                    onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])} />
                </div>
                {analyzing && (
                  <div style={{ textAlign: "center", padding: "16px 0" }}>
                    <div style={S.spinner} />
                    <div style={{ color: "#38bdf8", fontFamily: "'DM Mono',monospace", fontSize: 12, marginTop: 10 }}>
                      RUNNING 3-STAGE FORENSIC ANALYSIS…
                    </div>
                  </div>
                )}
              </div>

              <div style={S.pageCard}>
                <div style={S.panelHeader}><span style={S.panelTitle}>Analysis Result</span></div>
                {analysisResult ? (
                  <div>
                    <div style={{
                      background: analysisResult.isFake ? "rgba(240,75,90,0.1)" : "rgba(45,212,160,0.1)",
                      border: `1px solid ${analysisResult.isFake ? "#f04b5a" : "#2dd4a0"}40`,
                      borderRadius: 8, padding: "16px", marginBottom: 16, textAlign: "center"
                    }}>
                      <div style={{ fontSize: 36, marginBottom: 4 }}>{analysisResult.isFake ? "🚨" : "✅"}</div>
                      <div style={{
                        fontFamily: "'DM Mono',monospace", fontSize: 22, fontWeight: 700,
                        color: analysisResult.isFake ? "#f04b5a" : "#2dd4a0"
                      }}>
                        {analysisResult.isFake ? "DEEPFAKE DETECTED" : "AUTHENTIC IMAGE"}
                      </div>
                      <div style={{ color: "#64748b", fontSize: 13, marginTop: 4 }}>
                        Confidence: {analysisResult.confidence}%
                      </div>
                    </div>
                    {[
                      { label: "Face Manipulation",       val: analysisResult.faceManipulation,      color: "#f04b5a" },
                      { label: "Deepfake Artifacts",      val: analysisResult.deepfakeArtifacts,     color: "#f0a84b" },
                      { label: "AI Generated Indicators", val: analysisResult.aiGeneratedIndicators, color: "#a78bfa" },
                    ].map((m) => (
                      <div key={m.label} style={{ marginBottom: 12 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ fontSize: 12, color: "#94a3b8" }}>{m.label}</span>
                          <span style={{ fontSize: 12, color: m.color, fontFamily: "'DM Mono',monospace", fontWeight: 600 }}>
                            {m.val}%
                          </span>
                        </div>
                        <div style={{ height: 6, background: "#0f172a", borderRadius: 3 }}>
                          <div style={{ width: `${m.val}%`, height: "100%", background: m.color, borderRadius: 3, transition: "width 1s ease" }} />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ textAlign: "center", padding: "40px 0", color: "#4a5568", fontFamily: "'DM Mono',monospace", fontSize: 12 }}>
                    AWAITING IMAGE UPLOAD
                  </div>
                )}
              </div>
            </div>

            {/* Active queue */}
            <div style={{ ...S.pageCard, marginTop: 16 }}>
              <div style={S.panelHeader}>
                <span style={S.panelTitle}>Active Analysis Queue</span>
                <span style={{ fontSize: 12, color: "#f0a84b", fontFamily: "'DM Mono',monospace" }}>
                  {liveCount} In Progress
                </span>
              </div>
              {[...Array(4)].map((_, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 0", borderBottom: "1px solid #0f172a" }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#f0a84b", boxShadow: "0 0 6px #f0a84b", animation: "pulse 1.5s infinite" }} />
                  <span style={{ flex: 1, fontSize: 12, color: "#94a3b8", fontFamily: "'DM Mono',monospace" }}>
                    scan_job_{String(4891 - i).padStart(4, "0")}.task
                  </span>
                  <span style={{ fontSize: 11, color: "#f0a84b" }}>PROCESSING</span>
                  <div style={{ width: 80, height: 4, background: "#0f172a", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{
                      width: `${[72, 45, 88, 31][i]}%`,
                      height: "100%", background: "#f0a84b", borderRadius: 2
                    }} />
                  </div>
                  <span style={{ fontSize: 11, color: "#64748b" }}>{[72, 45, 88, 31][i]}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── DETECTION LOGS ── */}
        {activeNav === "Detection Logs" && (
          <div style={S.content}>
            <div style={S.pageCard}>
              <div style={{ ...S.panelHeader, marginBottom: 16 }}>
                <span style={S.panelTitle}>Detection Log History</span>
                <div style={{ display: "flex", gap: 8 }}>
                  <select style={S.select}>
                    <option>All Results</option>
                    <option>Fake Only</option>
                    <option>Real Only</option>
                    <option>Under Review</option>
                  </select>
                  <button style={S.viewAllBtn}>Export CSV</button>
                </div>
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {["Log ID", "File Name", "Result", "Confidence", "Timestamp", "Action"].map((h) => (
                      <th key={h} style={S.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log) => (
                    <tr key={log.id} style={{ borderBottom: "1px solid #0f172a", transition: "background 0.2s" }}
                      onMouseEnter={(e) => e.currentTarget.style.background = "#0f172a"}
                      onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                    >
                      <td style={{ ...S.td, fontFamily: "'DM Mono',monospace", color: "#38bdf8" }}>{log.id}</td>
                      <td style={S.td}>{log.file}</td>
                      <td style={S.td}>
                        <span style={{
                          fontSize: 11, fontWeight: 700, fontFamily: "'DM Mono',monospace",
                          color: log.result === "FAKE" ? "#f04b5a" : log.result === "REAL" ? "#2dd4a0" : "#f0a84b",
                          background: log.result === "FAKE" ? "rgba(240,75,90,0.1)" : log.result === "REAL" ? "rgba(45,212,160,0.1)" : "rgba(240,168,75,0.1)",
                          padding: "2px 8px", borderRadius: 4,
                        }}>{log.result}</span>
                      </td>
                      <td style={{ ...S.td, fontFamily: "'DM Mono',monospace", color: "#94a3b8" }}>
                        {log.conf.toFixed(1)}%
                      </td>
                      <td style={{ ...S.td, color: "#4a5568", fontFamily: "'DM Mono',monospace", fontSize: 11 }}>{log.ts}</td>
                      <td style={S.td}>
                        <button style={{ ...S.viewAllBtn, fontSize: 11 }}>View</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 16 }}>
                <span style={{ fontSize: 12, color: "#4a5568" }}>Showing {logs.length} entries</span>
                <div style={{ display: "flex", gap: 6 }}>
                  {["← Prev", "1", "2", "3", "Next →"].map((b) => (
                    <button key={b} style={{ ...S.viewAllBtn, padding: "4px 10px", fontSize: 11 }}>{b}</button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── SYSTEM SETTINGS ── */}
        {activeNav === "System Settings" && (
          <div style={S.content}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {[
                {
                  title: "Detection Thresholds",
                  fields: [
                    { label: "Fake Detection Threshold", type: "range", value: 60, unit: "%" },
                    { label: "Confidence Minimum",        type: "range", value: 72, unit: "%" },
                    { label: "Spoof Detection Score",     type: "range", value: 45, unit: "%" },
                  ],
                },
                {
                  title: "Model Configuration",
                  fields: [
                    { label: "Primary Model",      type: "select", options: ["EfficientNet-B0", "ResNet-50", "VGG-16"] },
                    { label: "Image Input Size",    type: "select", options: ["224×224", "299×299", "512×512"] },
                    { label: "Batch Size",          type: "number", value: 16 },
                  ],
                },
                {
                  title: "Alert Configuration",
                  fields: [
                    { label: "Email Alerts",         type: "toggle", value: true  },
                    { label: "SMS Notifications",    type: "toggle", value: false },
                    { label: "Alert Threshold (%)",  type: "number", value: 85    },
                  ],
                },
                {
                  title: "System Preferences",
                  fields: [
                    { label: "Auto-purge Logs (days)", type: "number", value: 30 },
                    { label: "Max Upload Size (MB)",   type: "number", value: 500 },
                    { label: "Dark Mode",              type: "toggle", value: true },
                  ],
                },
              ].map((section) => (
                <div key={section.title} style={S.pageCard}>
                  <div style={{ ...S.panelTitle, marginBottom: 16, paddingBottom: 12, borderBottom: "1px solid #1e293b" }}>
                    {section.title}
                  </div>
                  {section.fields.map((f) => (
                    <div key={f.label} style={{ marginBottom: 14 }}>
                      <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                        <span style={{ fontSize: 12, color: "#94a3b8" }}>{f.label}</span>
                        {f.type === "toggle" && (
                          <div style={{
                            width: 40, height: 20, borderRadius: 10, cursor: "pointer",
                            background: f.value ? "#38bdf8" : "#1e293b",
                            position: "relative", transition: "background 0.3s",
                          }}>
                            <div style={{
                              position: "absolute", top: 2,
                              left: f.value ? 22 : 2,
                              width: 16, height: 16, borderRadius: "50%",
                              background: "#fff", transition: "left 0.3s",
                            }} />
                          </div>
                        )}
                      </label>
                      {f.type === "range" && (
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <input type="range" min="0" max="100" defaultValue={f.value}
                            style={{ flex: 1, accentColor: "#38bdf8" }} />
                          <span style={{ fontSize: 12, color: "#38bdf8", fontFamily: "'DM Mono',monospace", width: 36 }}>
                            {f.value}{f.unit}
                          </span>
                        </div>
                      )}
                      {f.type === "select" && (
                        <select style={{ ...S.select, width: "100%" }}>
                          {f.options.map((o) => <option key={o}>{o}</option>)}
                        </select>
                      )}
                      {f.type === "number" && (
                        <input type="number" defaultValue={f.value}
                          style={{ ...S.select, width: "100%", padding: "6px 10px" }} />
                      )}
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 10, marginTop: 16 }}>
              <button style={{ ...S.viewAllBtn, padding: "8px 24px" }}>Reset Defaults</button>
              <button
                style={{
                  padding: "8px 24px",
                  background: "linear-gradient(135deg, #38bdf8, #3d8ef8)",
                  color: "#fff", border: "none", borderRadius: 6,
                  fontFamily: "'DM Mono',monospace", fontSize: 12,
                  cursor: "pointer", fontWeight: 600, letterSpacing: "1px",
                }}
                onClick={() => { setSettingsSaved(true); setTimeout(() => setSettingsSaved(false), 2000); }}
              >
                {settingsSaved ? "✓ SAVED!" : "SAVE SETTINGS"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* CSS animations */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; }
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes ringPulse {
          0%,100% { box-shadow: 0 0 0 0 rgba(56,189,248,0.4); }
          50% { box-shadow: 0 0 0 6px rgba(56,189,248,0); }
        }
        input[type=range]::-webkit-slider-thumb { background: #38bdf8; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: #080c14; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
      `}</style>
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────
const S = {
  root: {
    display: "flex",
    height: "100vh",
    background: "#080c14",
    color: "#e2e8f0",
    fontFamily: "'Syne', sans-serif",
    overflow: "hidden",
  },
  sidebar: {
    width: 210,
    minWidth: 210,
    background: "#0a0f1e",
    borderRight: "1px solid #0f172a",
    display: "flex",
    flexDirection: "column",
    padding: "0 0 16px",
    overflowY: "auto",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "18px 16px 20px",
    borderBottom: "1px solid #0f172a",
    marginBottom: 8,
  },
  logoMark: {
    width: 36, height: 36,
    background: "linear-gradient(135deg, #0ea5e9, #3d8ef8)",
    borderRadius: 8,
    display: "flex", alignItems: "center", justifyContent: "center",
    boxShadow: "0 0 16px rgba(56,189,248,0.4)",
  },
  logoText: {
    fontFamily: "'DM Mono', monospace",
    fontWeight: 500,
    fontSize: 11,
    letterSpacing: "1.5px",
    color: "#e2e8f0",
  },
  nav: { flex: 1, padding: "8px 8px" },
  navItem: {
    display: "flex", alignItems: "center", gap: 10,
    padding: "10px 12px",
    background: "transparent", border: "none", borderRadius: 8,
    color: "#64748b", cursor: "pointer", width: "100%",
    textAlign: "left", transition: "all 0.2s",
    fontFamily: "'Syne', sans-serif",
  },
  navItemActive: {
    background: "rgba(56,189,248,0.08)",
    color: "#e2e8f0",
    borderLeft: "2px solid #38bdf8",
    paddingLeft: 10,
  },
  sidebarFooter: {
    padding: "12px 16px",
    borderTop: "1px solid #0f172a",
    display: "flex", alignItems: "center", gap: 8,
  },
  pulseRing: {
    width: 8, height: 8, borderRadius: "50%",
    background: "#2dd4a0",
    boxShadow: "0 0 6px #2dd4a0",
    animation: "ringPulse 2s infinite",
  },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  topbar: {
    display: "flex", alignItems: "center",
    justifyContent: "space-between",
    padding: "0 20px",
    height: 54,
    background: "#0a0f1e",
    borderBottom: "1px solid #0f172a",
    flexShrink: 0,
  },
  pageTitle: {
    fontFamily: "'DM Mono', monospace",
    fontSize: 13,
    fontWeight: 500,
    color: "#38bdf8",
    letterSpacing: "2px",
    margin: 0,
  },
  topbarRight: { display: "flex", alignItems: "center", gap: 8 },
  topbarBtn: {
    display: "flex", alignItems: "center",
    background: "transparent", border: "1px solid #1e293b",
    borderRadius: 6, padding: "6px 12px",
    color: "#94a3b8", cursor: "pointer",
    fontFamily: "'Syne', sans-serif", fontSize: 12,
    transition: "all 0.2s", position: "relative",
  },
  badge: {
    position: "absolute", top: -4, right: -4,
    background: "#f04b5a", color: "#fff",
    fontSize: 9, fontWeight: 700,
    borderRadius: "50%", width: 16, height: 16,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontFamily: "'DM Mono',monospace",
  },
  adminChip: {
    display: "flex", alignItems: "center", gap: 8,
    background: "#0f172a", border: "1px solid #1e293b",
    borderRadius: 8, padding: "4px 12px",
    cursor: "pointer",
  },
  adminAvatar: {
    width: 26, height: 26, borderRadius: "50%",
    background: "linear-gradient(135deg, #38bdf8, #6366f1)",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 12, fontWeight: 700, color: "#fff",
  },
  content: {
    flex: 1,
    padding: "16px 20px",
    overflowY: "auto",
  },
  statGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    background: "#0d1321",
    border: "1px solid #1e293b",
    borderRadius: 10,
    padding: "14px 16px",
  },
  statLabel: { fontSize: 11, color: "#64748b", marginBottom: 4 },
  statValue: { fontSize: 28, fontWeight: 800, lineHeight: 1.1, marginBottom: 4 },
  statBadge: {
    fontSize: 10, fontFamily: "'DM Mono',monospace",
    padding: "2px 7px", borderRadius: 4, letterSpacing: "0.5px",
  },
  mainRow: {
    display: "grid",
    gridTemplateColumns: "1fr 340px",
    gap: 14,
  },
  analysisPanel: {
    background: "#0d1321",
    border: "1px solid #1e293b",
    borderRadius: 10,
    padding: "14px 16px",
    overflow: "hidden",
  },
  panelHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  panelTitle: {
    fontSize: 13,
    fontWeight: 700,
    color: "#38bdf8",
    letterSpacing: "0.5px",
  },
  imagePair: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 8,
    marginBottom: 10,
  },
  imageBox: {
    position: "relative",
    height: 200,
    background: "#080c14",
    border: "1px solid",
    borderRadius: 6,
    overflow: "hidden",
    cursor: "pointer",
  },
  imageBoxLabel: {
    position: "absolute",
    top: 8, left: 8,
    background: "rgba(0,0,0,0.7)",
    color: "#e2e8f0",
    fontSize: 10,
    fontFamily: "'DM Mono',monospace",
    letterSpacing: "1px",
    padding: "2px 8px",
    borderRadius: 3,
    zIndex: 2,
  },
  uploadPlaceholder: {
    height: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
  },
  analyzingOverlay: {
    position: "absolute", inset: 0,
    background: "rgba(8,12,20,0.75)",
    display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    zIndex: 3,
  },
  verdictOverlay: {
    position: "absolute",
    bottom: 0, left: 0, right: 0,
    textAlign: "center",
    padding: "6px",
    fontFamily: "'DM Mono',monospace",
    fontWeight: 700,
    fontSize: 13,
    letterSpacing: "2px",
    color: "#fff",
    zIndex: 2,
  },
  spinner: {
    width: 28, height: 28,
    border: "3px solid #1e293b",
    borderTop: "3px solid #38bdf8",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  probBar: {
    display: "flex",
    height: 8,
    borderRadius: 4,
    overflow: "hidden",
    marginBottom: 6,
    background: "#1e293b",
  },
  probLabels: {
    display: "flex",
    gap: 16,
    marginBottom: 10,
    fontSize: 12,
    fontFamily: "'DM Mono',monospace",
  },
  probDot: (color) => ({
    display: "inline-block",
    width: 8, height: 8,
    borderRadius: "50%",
    background: color,
    marginRight: 4,
  }),
  detailedBox: {
    background: "#080c14",
    border: "1px solid #1e293b",
    borderRadius: 6,
    padding: "12px",
    marginBottom: 4,
  },
  detailedTitle: {
    fontSize: 12, fontWeight: 700,
    color: "#94a3b8", marginBottom: 10,
    fontFamily: "'DM Mono',monospace",
    letterSpacing: "0.5px",
  },
  rightCol: { display: "flex", flexDirection: "column" },
  sidePanel: {
    background: "#0d1321",
    border: "1px solid #1e293b",
    borderRadius: 10,
    padding: "14px 16px",
  },
  alertRow: {
    display: "flex", alignItems: "center", gap: 8,
    padding: "9px 10px",
    marginBottom: 6,
    background: "#080c14",
    borderRadius: 6,
    cursor: "pointer",
    transition: "background 0.2s",
  },
  alertTime: {
    fontSize: 10, color: "#4a5568",
    fontFamily: "'DM Mono',monospace",
  },
  viewAllBtn: {
    background: "transparent",
    border: "1px solid #1e293b",
    borderRadius: 5,
    color: "#38bdf8",
    fontSize: 12,
    padding: "4px 12px",
    cursor: "pointer",
    fontFamily: "'DM Mono',monospace",
    letterSpacing: "0.5px",
    transition: "all 0.2s",
  },
  select: {
    background: "#080c14",
    border: "1px solid #1e293b",
    borderRadius: 5,
    color: "#94a3b8",
    fontSize: 12,
    padding: "5px 10px",
    cursor: "pointer",
    fontFamily: "'DM Mono',monospace",
    outline: "none",
  },
  pageCard: {
    background: "#0d1321",
    border: "1px solid #1e293b",
    borderRadius: 10,
    padding: "16px",
  },
  dropZone: {
    border: "2px dashed",
    borderRadius: 8,
    padding: "32px",
    textAlign: "center",
    cursor: "pointer",
    minHeight: 140,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.2s",
  },
  th: {
    padding: "10px 12px",
    fontSize: 11,
    color: "#4a5568",
    fontFamily: "'DM Mono',monospace",
    letterSpacing: "1px",
    textAlign: "left",
    background: "#080c14",
    borderBottom: "1px solid #0f172a",
    fontWeight: 500,
  },
  td: {
    padding: "10px 12px",
    fontSize: 12,
    color: "#94a3b8",
    fontFamily: "'Syne',sans-serif",
  },
};