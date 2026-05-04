**[→ Live demo](https://gabrielebettineschi.com/gaussian-kernel-plot)**

# Gaussian process regression

A Gaussian process regression starts with a prior distribution over functions and updates it with data, concentrating it on the functions that are probable given that data. 

This program plots the mean (a function) and some samples (some functions) of said distirbution. You can click on the canvas to place observations and watch the distribution update in real time.

##  Some explanations

The kernel of a gaussain process defines the class of functions that the process can represent. In this program, the exponential and the radial basis function (RBF) kernel are available. 

There are some parameters you can set, that control the typical function: amplitude sets the vertical scale, length scale sets how far orizontally values are correlated, noise controls how strictly values adhere to data.

**Calibrated uncertainty.** The shaded bands show ±1σ and ±2σ. This is the practical power of GPs: they always know what they don't know. Far from any observation the uncertainty is high; near observations it collapses. This makes GPs valuable in Bayesian optimization, scientific modeling, and engineering surrogates where data is scarce and overconfidence is costly.

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
