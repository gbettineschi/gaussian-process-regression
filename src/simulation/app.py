"""
Gaussian Process Regression — Pyodide web app.

GP math + canvas rendering in a single file.
Run in-browser via Pyodide (loaded by index.html).
For a local interactive version, see main.py.
"""

import math
import numpy as np
from js import document, window
from pyodide.ffi import create_proxy

# ── GP math ───────────────────────────────────────────────────────────────────

X_TEST   = np.linspace(0, 1, 200)
N_SAMPLES = 5


def rbf_kernel(x1, x2, amplitude, length_scale):
    d = (x1 - x2) / length_scale
    return amplitude * np.exp(-0.5 * d ** 2)


def exp_kernel(x1, x2, amplitude, length_scale):
    return amplitude * np.exp(-np.abs(x1 - x2) / length_scale)


def _kernel_matrix(X1, X2, kernel_fn, amplitude, length_scale):
    X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
    X2 = np.asarray(X2, dtype=float).reshape(1, -1)
    return kernel_fn(X1, X2, amplitude, length_scale)


def gp_compute(X_train, y_train, kernel_fn, amplitude, length_scale, noise,
               n_samples=N_SAMPLES):
    """
    Returns {'mean': array, 'std': array, 'samples': list[array]}.

    If X_train is empty, returns the GP prior.
    Otherwise conditions on (X_train, y_train) to produce the posterior.
    """
    m = len(X_TEST)

    K_mm = _kernel_matrix(X_TEST, X_TEST, kernel_fn, amplitude, length_scale)
    np.fill_diagonal(K_mm, K_mm.diagonal() + 1e-8)

    if len(X_train) == 0:
        mean = np.zeros(m)
        cov  = K_mm
    else:
        X_tr = np.asarray(X_train, dtype=float)
        y_tr = np.asarray(y_train, dtype=float)
        n    = len(X_tr)

        K_nn  = _kernel_matrix(X_tr, X_tr, kernel_fn, amplitude, length_scale)
        K_nn += (noise ** 2 + 1e-8) * np.eye(n)
        K_mn  = _kernel_matrix(X_TEST, X_tr, kernel_fn, amplitude, length_scale)

        # mean* = K_mn @ K_nn^{-1} @ y
        alpha = np.linalg.solve(K_nn, y_tr)
        mean  = K_mn @ alpha

        # cov* = K_mm − K_mn @ K_nn^{-1} @ K_mn^T
        V   = np.linalg.solve(K_nn, K_mn.T)
        cov = K_mm - K_mn @ V
        np.fill_diagonal(cov, np.maximum(np.diag(cov), 0.0) + 1e-8)

    std = np.sqrt(np.diag(cov))

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += 1e-6 * np.eye(m)
        L    = np.linalg.cholesky(cov)

    samples = [mean + L @ np.random.randn(m) for _ in range(n_samples)]

    return {"mean": mean, "std": std, "samples": samples}


KERNELS = {"rbf": rbf_kernel, "exp": exp_kernel}


# ── Canvas app ────────────────────────────────────────────────────────────────

Y_MIN, Y_MAX  = -3.5, 3.5
PAD           = 16        # uniform padding (CSS px)
HIT_RADIUS    = 14        # click/drag detection radius (CSS px)

observations  = []        # [[x, y], ...] in GP coords
kernel_name   = "rbf"
amplitude     = 1.0
length_scale  = 0.3
noise         = 0.05
dragging      = None      # index into observations, or None

result        = None
canvas        = None
ctx           = None
disp_w        = 1.0
disp_h        = 1.0
_proxies      = []        # keep create_proxy alive against GC


def _keep(fn):
    p = create_proxy(fn)
    _proxies.append(p)
    return p


# ── Coordinate transforms ──────────────────────────────────────────────────────

def to_pixel(gx, gy):
    pw = disp_w - 2 * PAD
    ph = disp_h - 2 * PAD
    px = PAD + gx * pw
    py = PAD + (1.0 - (gy - Y_MIN) / (Y_MAX - Y_MIN)) * ph
    return px, py


def to_gp(px, py):
    pw = disp_w - 2 * PAD
    ph = disp_h - 2 * PAD
    gx = (px - PAD) / pw
    gy = Y_MIN + (1.0 - (py - PAD) / ph) * (Y_MAX - Y_MIN)
    return max(0.0, min(1.0, gx)), max(Y_MIN, min(Y_MAX, gy))


# ── Compute & render ──────────────────────────────────────────────────────────

def compute():
    global result
    result = gp_compute(
        [o[0] for o in observations],
        [o[1] for o in observations],
        KERNELS[kernel_name],
        amplitude, length_scale, noise,
    )


def render():
    if result is None:
        return
    mean    = result["mean"]
    std     = result["std"]
    samples = result["samples"]
    dpr     = window.devicePixelRatio or 1.0
    m       = len(X_TEST)

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.fillStyle = "#0d0d1a"
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    # Zero line
    _, zy = to_pixel(0, 0)
    ctx.beginPath()
    ctx.moveTo(PAD * dpr, zy * dpr)
    ctx.lineTo((disp_w - PAD) * dpr, zy * dpr)
    ctx.strokeStyle = "rgba(255,255,255,0.06)"
    ctx.lineWidth = dpr
    ctx.stroke()

    def fill_band(upper, lower, color):
        ctx.beginPath()
        for i in range(m):
            px, py = to_pixel(float(X_TEST[i]), float(upper[i]))
            if i == 0:
                ctx.moveTo(px * dpr, py * dpr)
            else:
                ctx.lineTo(px * dpr, py * dpr)
        for i in range(m - 1, -1, -1):
            px, py = to_pixel(float(X_TEST[i]), float(lower[i]))
            ctx.lineTo(px * dpr, py * dpr)
        ctx.closePath()
        ctx.fillStyle = color
        ctx.fill()

    fill_band(mean + 2 * std, mean - 2 * std, "rgba(100,149,237,0.13)")
    fill_band(mean + std,     mean - std,     "rgba(100,149,237,0.10)")

    # Sample paths
    ctx.lineWidth = dpr
    for sample in samples:
        ctx.beginPath()
        for i in range(m):
            px, py = to_pixel(float(X_TEST[i]), float(sample[i]))
            if i == 0:
                ctx.moveTo(px * dpr, py * dpr)
            else:
                ctx.lineTo(px * dpr, py * dpr)
        ctx.strokeStyle = "rgba(120,165,255,0.28)"
        ctx.stroke()

    # Mean line
    ctx.beginPath()
    for i in range(m):
        px, py = to_pixel(float(X_TEST[i]), float(mean[i]))
        if i == 0:
            ctx.moveTo(px * dpr, py * dpr)
        else:
            ctx.lineTo(px * dpr, py * dpr)
    ctx.strokeStyle = "#7eaaff"
    ctx.lineWidth = 2.0 * dpr
    ctx.stroke()

    # Observation points
    for obs in observations:
        px, py = to_pixel(obs[0], obs[1])
        ctx.beginPath()
        ctx.arc(px * dpr, py * dpr, 6.0 * dpr, 0, 2 * math.pi)
        ctx.fillStyle = "#ff7070"
        ctx.fill()
        ctx.strokeStyle = "#ffaaaa"
        ctx.lineWidth = 1.5 * dpr
        ctx.stroke()

    if not observations:
        ctx.font = f"{int(13 * dpr)}px system-ui, sans-serif"
        ctx.fillStyle = "rgba(180,180,220,0.35)"
        ctx.textAlign = "center"
        ctx.fillText(
            "Click anywhere to place an observation",
            disp_w / 2.0 * dpr,
            (disp_h - 24) * dpr,
        )


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_canvas(_event=None):
    global disp_w, disp_h
    dpr           = window.devicePixelRatio or 1.0
    rect          = canvas.parentElement.getBoundingClientRect()
    disp_w        = rect.width
    disp_h        = rect.height
    canvas.width  = disp_w * dpr
    canvas.height = disp_h * dpr
    canvas.style.width  = f"{disp_w}px"
    canvas.style.height = f"{disp_h}px"
    render()


# ── Interaction helpers ───────────────────────────────────────────────────────

def _event_pos(event):
    rect = canvas.getBoundingClientRect()
    return event.clientX - rect.left, event.clientY - rect.top


def _nearest(px, py):
    best_i, best_d = -1, float("inf")
    for i, obs in enumerate(observations):
        ox, oy = to_pixel(obs[0], obs[1])
        d = math.hypot(px - ox, py - oy)
        if d < best_d:
            best_d, best_i = d, i
    return best_i if best_d <= HIT_RADIUS else -1


def _remove_nearest(px, py):
    idx = _nearest(px, py)
    if idx != -1:
        observations.pop(idx)
        compute()
        render()


# ── Event handlers ────────────────────────────────────────────────────────────

def on_mousedown(event):
    global dragging
    if event.button == 2:
        return
    event.preventDefault()
    px, py = _event_pos(event)
    if event.altKey:
        _remove_nearest(px, py)
        return
    idx = _nearest(px, py)
    if idx != -1:
        dragging = idx
    else:
        gx, gy = to_gp(px, py)
        observations.append([gx, gy])
        dragging = len(observations) - 1
        compute()
        render()


def on_mousemove(event):
    if dragging is None:
        return
    event.preventDefault()
    px, py = _event_pos(event)
    gx, gy = to_gp(px, py)
    observations[dragging][0] = gx
    observations[dragging][1] = gy
    compute()
    render()


def on_mouseup(_event):
    global dragging
    dragging = None


def on_dblclick(event):
    px, py = _event_pos(event)
    _remove_nearest(px, py)


def on_contextmenu(event):
    event.preventDefault()
    px, py = _event_pos(event)
    _remove_nearest(px, py)


def on_kernel_click(event):
    global kernel_name
    kernel_name = event.currentTarget.getAttribute("data-kernel")
    for b in document.querySelectorAll(".kernel-btn"):
        b.classList.remove("active")
    event.currentTarget.classList.add("active")
    compute()
    render()


def on_amplitude_input(event):
    global amplitude
    amplitude = float(event.target.value)
    document.getElementById("val-amplitude").textContent = f"{amplitude:.2f}"
    compute()
    render()


def on_lengthscale_input(event):
    global length_scale
    length_scale = float(event.target.value)
    document.getElementById("val-lengthscale").textContent = f"{length_scale:.2f}"
    compute()
    render()


def on_noise_input(event):
    global noise
    noise = float(event.target.value)
    document.getElementById("val-noise").textContent = f"{noise:.2f}"
    compute()
    render()


def on_clear(_event):
    observations.clear()
    compute()
    render()


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def init():
    global canvas, ctx
    canvas = document.getElementById("gp-canvas")
    ctx    = canvas.getContext("2d")

    for btn in document.querySelectorAll(".kernel-btn"):
        btn.addEventListener("click", _keep(on_kernel_click))

    document.getElementById("slider-amplitude").addEventListener("input",   _keep(on_amplitude_input))
    document.getElementById("slider-lengthscale").addEventListener("input", _keep(on_lengthscale_input))
    document.getElementById("slider-noise").addEventListener("input",       _keep(on_noise_input))
    document.getElementById("btn-clear").addEventListener("click",          _keep(on_clear))

    canvas.addEventListener("mousedown",   _keep(on_mousedown))
    canvas.addEventListener("mousemove",   _keep(on_mousemove))
    canvas.addEventListener("mouseup",     _keep(on_mouseup))
    canvas.addEventListener("dblclick",    _keep(on_dblclick))
    canvas.addEventListener("contextmenu", _keep(on_contextmenu))
    window.addEventListener("resize",      _keep(resize_canvas))

    resize_canvas()
    compute()
    render()


init()
