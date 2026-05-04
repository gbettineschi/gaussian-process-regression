# Gaussian Process Regression

An interactive visualization of Bayesian inference over functions.

**[Live demo →](https://gabrielebettineschi.com/gaussian-kernel-plot)**

---

## What is a Gaussian Process?

Most machine learning models output a single prediction. A Gaussian Process outputs a *distribution* — not over numbers, but over **entire functions**. Before you show it any data, it holds every plausible function in mind simultaneously. When you give it observations, it updates, narrowing the distribution to only those functions that are consistent with what you've seen.

This is Bayesian inference, but at the level of function space.

---

## The kernel defines what "plausible" means

The **kernel** (or covariance function) encodes prior beliefs about the shape of the unknown function:

| Kernel | Formula | Character |
|--------|---------|-----------|
| **RBF** (Radial Basis Function) | `k(x,x') = a · exp(-(x-x')² / 2l²)` | Infinitely smooth — nearby points are strongly correlated |
| **Exponential** | `k(x,x') = a · exp(-\|x-x'\| / l)` | Rough — only once-differentiable, slight kinks at observations |

**Amplitude** `a` controls the overall variance — how far functions stray from zero.  
**Length scale** `l` controls smoothness — small `l` gives wiggly functions, large `l` gives slowly varying ones.

---

## Bayesian updating in real time

When you click to place an observation, you are conditioning the GP on a new fact: *"the true function passes through this point."* The posterior distribution is computed analytically:

```
mean*  = K(X*, X) · [K(X,X) + σ²I]⁻¹ · y
cov*   = K(X*,X*) − K(X*,X) · [K(X,X) + σ²I]⁻¹ · K(X,X*)
```

Watch how the uncertainty band (±2σ) **collapses near observations** and **expands where you haven't looked**.

---

## Uncertainty quantification

This is the practical power of GPs: they always know what they don't know. Far from any observation, the uncertainty is high — the model admits it could be wrong. This makes GPs valuable in:

- **Scientific modeling** where data is expensive and confidence matters
- **Bayesian optimization** (finding the optimum of an unknown function efficiently)
- **Surrogate modeling** in engineering simulations

The **noise** slider controls the assumed observation noise σ — high noise means the GP won't try to pass exactly through your points, resulting in smoother but less committed fits.

---

## Usage

Open `index.html` directly in a browser, or serve it locally:

```bash
python3 -m http.server
```

| Action | Effect |
|--------|--------|
| Click on canvas | Add observation |
| Drag a point | Move observation |
| Right-click / double-click | Remove observation |
| Clear button | Reset to prior |

---

## Implementation

The entire GP runs in vanilla JavaScript with no external dependencies:

- **`src/js/matrix.js`** — Cholesky decomposition, forward/backward substitution, dot products
- **`src/js/kernels.js`** — RBF and Exponential kernel functions + kernel matrix construction
- **`src/js/gp.js`** — Prior sampling and posterior computation (mean, variance, sample paths)
- **`src/js/ui.js`** — Canvas rendering, interaction handling, responsive layout

The Python reference implementation (NumPy + Matplotlib) lives in `main.py`.
