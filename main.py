import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def cov_matrix(t, a, b, kind):
    # t: (n,)
    dt = t[:, None] - t[None, :]
    if kind == "exp":
        K = a * np.exp(-b * np.abs(dt))
    elif kind == "rbf":
        K = a * np.exp(-b * (dt ** 2))
    else:
        raise ValueError("kind must be 'exp' or 'rbf'")
    return K

def sample_gp(t, a, b, kind, n_samples=10, jitter=1e-10, seed=0):
    rng = np.random.default_rng(seed)
    K = cov_matrix(t, a, b, kind)
    # numerical stability
    K = K + jitter * np.eye(len(t))
    L = np.linalg.cholesky(K)
    Z = rng.standard_normal((len(t), n_samples))
    return L @ Z  # (n, n_samples)

# domain
t = np.linspace(0, 1, 250)

# initial params
a0, b0 = 1.0, 10.0
kind0 = "exp"

fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(left=0.1, bottom=0.25, right=0.85)

lines = []

def redraw(a, b, kind):
    global lines
    # clear old lines
    for ln in lines:
        ln.remove()
    lines = []

    Y = sample_gp(t, a, b, kind, n_samples=10, seed=1)
    for i in range(Y.shape[1]):
        (ln,) = ax.plot(t, Y[:, i], linewidth=1)
        lines.append(ln)

    ax.set_title(f"GP prior samples | kernel={kind} | a={a:.3g}, b={b:.3g}")
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Sliders
ax_a = plt.axes([0.1, 0.12, 0.7, 0.03])
ax_b = plt.axes([0.1, 0.07, 0.7, 0.03])

s_a = Slider(ax_a, "a (variance)", valmin=0.01, valmax=5.0, valinit=a0)
s_b = Slider(ax_b, "b (1/length)", valmin=0.1, valmax=50.0, valinit=b0)

# Radio buttons for kernel
ax_radio = plt.axes([0.88, 0.55, 0.10, 0.15])
radio = RadioButtons(ax_radio, ("exp", "rbf"), active=0)

def on_change(_):
    redraw(s_a.val, s_b.val, radio.value_selected)

s_a.on_changed(on_change)
s_b.on_changed(on_change)
radio.on_clicked(on_change)

# initial draw
redraw(a0, b0, kind0)
plt.show()