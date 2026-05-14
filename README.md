**[→ Live demo](https://gabrielebettineschi.com/gaussian-kernel-plot)**

# Gaussian process regression

A Gaussian process regression starts with a prior distribution over functions and updates it with data, concentrating it on the functions that are probable given that data. 

This program plots the mean (a function) and some samples (some functions) of said distirbution. You can click on the canvas to place observations and watch the distribution update in real time.

## Run locally

**In the browser**

Serve `src/` with a local HTTP server. The first load fetches ~100 MB of Pyodide (Python compiled to WebAssembly); the service worker caches it for offline use afterward.

```bash
python3 -m http.server
# open http://localhost:8000/src/index.html
```

**With Python**

Install dependencies and run `main.py` directly. Visualization is through Matplotlib with sliders and radio buttons for kernel/parameter control.

```bash
uv sync
uv run main.py
```
