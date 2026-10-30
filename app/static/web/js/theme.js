// Applies the saved theme before the page paints (the CSP forbids inline scripts, so this is a tiny file).
try {
  const t = localStorage.getItem("jarvis.theme");
  if (t === "dark" || t === "light") document.documentElement.setAttribute("data-theme", t);
} catch { /* storage blocked: follow the system */ }
