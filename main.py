"""
Gaussian Process Regression — local interactive demo.

Click the plot to place observations and watch the posterior update in real time.
Right-click an observation to remove it.
Use the sliders and radio buttons to change kernel and parameters.

Run with:  uv run main.py   or   python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

# ── GP math ───────────────────────────────────────────────────────────────────

X_TEST = np.linspace(0, 1, 300)


def rbf_kernel(x1, x2, amplitude, length_scale):
    d = (x1 - x2) / length_scale
    return amplitude * np.exp(-0.5 * d ** 2)


def exp_kernel(x1, x2, amplitude, length_scale):
    return amplitude * np.exp(-np.abs(x1 - x2) / length_scale)


def kernel_matrix(X1, X2, kernel_fn, amplitude, length_scale):
    X1 = np.asarray(X1, dtype=float).reshape(-1, 1)
    X2 = np.asarray(X2, dtype=float).reshape(1, -1)
    return kernel_fn(X1, X2, amplitude, length_scale)


def gp_compute(X_train, y_train, kernel_fn, amplitude, length_scale, noise, n_samples=5):
    m = len(X_TEST)

    K_mm = kernel_matrix(X_TEST, X_TEST, kernel_fn, amplitude, length_scale)
    np.fill_diagonal(K_mm, K_mm.diagonal() + 1e-8)

    if len(X_train) == 0:
        mean = np.zeros(m)
        cov  = K_mm
    else:
        X_tr = np.asarray(X_train, dtype=float)
        y_tr = np.asarray(y_train, dtype=float)
        K_nn  = kernel_matrix(X_tr, X_tr, kernel_fn, amplitude, length_scale)
        K_nn += (noise ** 2 + 1e-8) * np.eye(len(X_tr))
        K_mn  = kernel_matrix(X_TEST, X_tr, kernel_fn, amplitude, length_scale)

        alpha = np.linalg.solve(K_nn, y_tr)
        mean  = K_mn @ alpha

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
    return mean, std, samples


KERNELS = {"RBF": rbf_kernel, "Exp": exp_kernel}

# ── State ─────────────────────────────────────────────────────────────────────

obs_x, obs_y = [], []

# ── Plot setup ────────────────────────────────────────────────────────────────

plt.style.use("dark_background")
fig = plt.figure(figsize=(11, 6))
fig.patch.set_facecolor("#0d0d1a")

ax = fig.add_axes([0.06, 0.18, 0.70, 0.75])
ax.set_facecolor("#12121f")
ax.set_xlim(0, 1)
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel("x", color="#8888aa")
ax.set_ylabel("f(x)", color="#8888aa")
ax.tick_params(colors="#555577")
for spine in ax.spines.values():
    spine.set_edgecolor("#222233")

# Plot objects (updated in place)
band2,  = ax.fill([], [], color="#6495ed", alpha=0.13)
band1,  = ax.fill([], [], color="#6495ed", alpha=0.10)
sample_lines = [ax.plot([], [], color="#78a5ff", alpha=0.28, lw=0.9)[0] for _ in range(5)]
mean_line,   = ax.plot([], [], color="#7eaaff", lw=2.0, zorder=3)
obs_scatter  = ax.scatter([], [], color="#ff7070", s=60, zorder=5, edgecolors="#ffaaaa", linewidths=1.2)

ax.axhline(0, color="white", alpha=0.05, lw=0.8)

# ── Widgets ───────────────────────────────────────────────────────────────────

ax_amp  = fig.add_axes([0.06, 0.10, 0.44, 0.03])
ax_ls   = fig.add_axes([0.06, 0.05, 0.44, 0.03])
ax_noise= fig.add_axes([0.06, 0.00, 0.44, 0.03])

s_amp   = Slider(ax_amp,   "Amplitude",    0.1,  3.0, valinit=1.0,  color="#6495ed")
s_ls    = Slider(ax_ls,    "Length scale", 0.05, 1.0, valinit=0.3,  color="#6495ed")
s_noise = Slider(ax_noise, "Noise",        0.01, 0.8, valinit=0.05, color="#6495ed")

ax_radio  = fig.add_axes([0.80, 0.55, 0.12, 0.18])
radio     = RadioButtons(ax_radio, list(KERNELS.keys()), active=0,
                         activecolor="#7eaaff")

ax_clear  = fig.add_axes([0.80, 0.42, 0.12, 0.07])
btn_clear = Button(ax_clear, "Clear", color="#1a1a2e", hovercolor="#2a1a2e")
btn_clear.label.set_color("#ff9090")

for widget_ax in [ax_amp, ax_ls, ax_noise, ax_radio, ax_clear]:
    widget_ax.set_facecolor("#13132a")

# ── Draw ──────────────────────────────────────────────────────────────────────

def redraw():
    kernel_fn    = KERNELS[radio.value_selected]
    amp          = s_amp.val
    ls           = s_ls.val
    n            = s_noise.val

    mean, std, samples = gp_compute(obs_x, obs_y, kernel_fn, amp, ls, n)

    upper2 = mean + 2 * std
    lower2 = mean - 2 * std
    upper1 = mean + std
    lower1 = mean - std

    xs = np.concatenate([X_TEST, X_TEST[::-1]])
    band2.set_xy(np.column_stack([xs, np.concatenate([upper2, lower2[::-1]])]))
    band1.set_xy(np.column_stack([xs, np.concatenate([upper1, lower1[::-1]])]))

    mean_line.set_data(X_TEST, mean)
    for ln, sample in zip(sample_lines, samples):
        ln.set_data(X_TEST, sample)

    if obs_x:
        obs_scatter.set_offsets(np.column_stack([obs_x, obs_y]))
    else:
        obs_scatter.set_offsets(np.empty((0, 2)))

    kernel_name = radio.value_selected
    ax.set_title(
        f"GP Regression  |  kernel={kernel_name}  a={amp:.2f}  l={ls:.2f}  σ={n:.2f}",
        color="#c8cce8", fontsize=10, pad=6,
    )
    fig.canvas.draw_idle()


# ── Events ────────────────────────────────────────────────────────────────────

def on_click(event):
    if event.inaxes is not ax:
        return
    if event.button == 1:
        obs_x.append(event.xdata)
        obs_y.append(event.ydata)
    elif event.button == 3:
        if not obs_x:
            return
        dists = [(event.xdata - x) ** 2 + (event.ydata - y) ** 2 for x, y in zip(obs_x, obs_y)]
        idx   = int(np.argmin(dists))
        obs_x.pop(idx)
        obs_y.pop(idx)
    redraw()


def on_clear(_event):
    obs_x.clear()
    obs_y.clear()
    redraw()


fig.canvas.mpl_connect("button_press_event", on_click)
btn_clear.on_clicked(on_clear)
s_amp.on_changed(lambda _: redraw())
s_ls.on_changed(lambda _: redraw())
s_noise.on_changed(lambda _: redraw())
radio.on_clicked(lambda _: redraw())

# ── Run ───────────────────────────────────────────────────────────────────────

redraw()
plt.show()
