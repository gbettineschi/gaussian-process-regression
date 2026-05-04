**[→ Live demo](https://gabrielebettineschi.com/gaussian-kernel-plot)**

# Gaussian Process Regression

Most models give you a single prediction. A Gaussian Process gives you a *distribution over functions* — before any data, it holds every plausible curve in mind at once. Click to place observations and watch it update in real time: uncertainty collapses where you've looked, and expands where you haven't.

## Theory

**Prior over functions.** A GP is fully defined by a mean function (assumed zero here) and a kernel. The kernel k(x, x') encodes how correlated two function values should be — it is the model's prior belief about the shape of the unknown function.

Two kernels are available:

| Kernel | Formula | Character |
|--------|---------|-----------|
| RBF | `k(x,x') = a · exp(-(x-x')² / 2l²)` | Infinitely smooth |
| Exponential | `k(x,x') = a · exp(-\|x-x'\| / l)` | Rougher, slight kinks at observations |

Amplitude `a` sets the vertical scale; length scale `l` controls wiggliness — small `l` gives rapidly varying functions, large `l` gives slowly varying ones.

**Bayesian update.** When you place an observation, the GP conditions on it analytically. Given training points X with values y and test points X*, the posterior mean and covariance are:

```
μ*  = K(X*, X) · [K(X, X) + σ²I]⁻¹ · y
Σ*  = K(X*, X*) − K(X*, X) · [K(X, X) + σ²I]⁻¹ · K(X, X*)
```

The noise parameter σ controls how strictly the model commits to passing through your points. Low noise → tight interpolation; high noise → smooth regression that trades fit for certainty.

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

## Controls

| Action | Effect |
|--------|--------|
| Click canvas | Add observation |
| Drag a point | Move observation |
| Right-click / double-click | Remove observation |
| Clear button | Reset to prior |
| Kernel toggle | Switch RBF ↔ Exponential |
| Amplitude / Length scale | Adjust kernel shape |
| Noise slider | Set observation noise σ |
