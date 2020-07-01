"""
Signals Generators
==================

This example shows the signal generators with a consistent function interface.
"""

import matplotlib.pyplot as plt
from lmlib.utils.generator import *

K = 5000
k_period = 500
y_sin = gen_sinusoidal(K, k_period)
y_rec = gen_rectangle(K, k_period, k_on=200)
y_ramp = gen_ramp(K, k_period)
y_tri = gen_triangle(K, k_period)
y_ui = gen_unit_impulse(K, k=300)
y_bl = gen_baseline_sin(K, k_period)
y_exp = gen_exponential(K, decay=1e-2, k=0)

signals = [y_sin, y_rec, y_ramp, y_tri, y_ui, y_bl, y_exp]
func_names = [
    "gen_sinusoidal",
    "gen_rectangle",
    "gen_ramp",
    "gen_triangle",
    "gen_unit_impulse",
    "gen_baseline_sin",
    "gen_exponential",
]
fig1, axs = plt.subplots(len(signals), 1, figsize=(6, 8))
for num, sig in enumerate(signals):
    axs[num].plot(range(K), sig, lw=0.8, c="k")
    axs[num].set_title(func_names[num])
plt.subplots_adjust(hspace=0.85)
plt.show()

seed = 1000
y_wgn = gen_wgn(K, sigma=0.3, seed=seed)
y_rramp = gen_rand_ramps(K, N=5, seed=seed)
y_rui = gen_rand_unit_impulse(K, N=3, seed=seed)
y_rpulse = gen_rand_pulse(K, N=3, length=100, seed=seed)
y_rwalk = gen_rand_walk(K, seed=seed)

signals = [y_wgn, y_rramp, y_rui, y_rpulse, y_rwalk]
fig2, axs = plt.subplots(len(signals), 1, figsize=(6, 8))
func_names = [
    "gen_wgn",
    "gen_rand_ramps",
    "gen_rand_unit_impulse",
    "gen_rand_pulse",
    "gen_rand_walk",
]
for num, sig in enumerate(signals):
    axs[num].plot(range(K), sig, lw=0.8, c="k")
    axs[num].set_title(func_names[num])

plt.subplots_adjust(hspace=0.85)
plt.show()


y_sin = gen_sinusoidal(K=100, k_period=100)
y_ui = gen_unit_impulse(K, k=[300, 1500, 4000])
y_conv = gen_convolve(y_ui, y_sin)

fig3, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(range(K), y_conv, lw=0.8, c="k")
ax.set_title("gen_convolve")
plt.subplots_adjust(hspace=0.5)
plt.show()
