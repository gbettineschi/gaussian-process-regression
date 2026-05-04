import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.mjs';

// Register service worker for Pyodide CDN caching
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./ui/sw.js').catch(() => {});
}

async function boot() {
  const status = document.getElementById('loading-status');

  const pyodide = await loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/',
  });

  status.textContent = 'Loading NumPy…';
  await pyodide.loadPackage('numpy');

  status.textContent = 'Starting GP engine…';
  const code = await fetch('./simulation/app.py').then(r => r.text());
  await pyodide.runPythonAsync(code);

  document.getElementById('loading').style.display = 'none';
}

boot().catch(err => {
  document.getElementById('loading-status').textContent = 'Error: ' + err.message;
  document.getElementById('loading-spinner').style.display = 'none';
  console.error(err);
});
