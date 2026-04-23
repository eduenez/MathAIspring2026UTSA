#!/usr/bin/env python3
"""
Generate three advanced-topic notebooks for MAT 4953/6973:
  code/diffusion_models.ipynb
  code/reinforcement_learning.ipynb
  code/geometric_deep_learning.ipynb
"""
import json, itertools, pathlib

_ctr = itertools.count(1)

def mk_id(): return f"c{next(_ctr):06d}"

def md(src):
    lines = src.strip('\n').split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "id": mk_id(), "metadata": {}, "source": source}

def code(src):
    lines = src.strip('\n').split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "id": mk_id(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

def notebook(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

# ════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — diffusion_models.ipynb
# ════════════════════════════════════════════════════════════════════════════

D = []

D.append(md(r"""# Diffusion Models and Score-Based Generative Modeling
## From Gaussian noise to data — and back

**MAT 4953/6973 — Mathematical Foundations of AI** (Spring 2026, UTSA)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/diffusion_models.ipynb)

---"""))

D.append(md(r"""> **Prerequisites:** [`generative_models.ipynb`](generative_models.ipynb) (VAEs, ELBO, reparameterization), probability (Gaussian distributions, Markov chains), basic calculus. Familiarity with the forward pass and backpropagation is helpful but not required for this overview.

In [`generative_models.ipynb`](generative_models.ipynb) we saw two generative paradigms: VAEs (encode to a latent, decode probabilistically) and autoregressive models (factor $p(\mathbf{x}) = \prod_t p(x_t | x_{<t})$). Both have limitations — VAEs tend to produce blurry samples; autoregressive models are slow at generation time.

**Diffusion models** (Ho et al., 2020; Song & Ermon, 2019) are currently the dominant paradigm for high-quality generation (Stable Diffusion, DALL·E, Sora). Their key idea is deceptively simple: *learn to reverse a noise-adding process*. The mathematics is elegant, drawing on Markov chains, Gaussian statistics, and the theory of stochastic differential equations.

**Outline**
1. [The forward diffusion process](#part1)
2. [Score functions](#part2)
3. [The reverse process and DDPM](#part3)
4. [Training: denoising score matching](#part4)
5. [Extensions: from discrete to continuous time](#part5)"""))

D.append(code(r"""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ipywidgets import interact, IntSlider, FloatSlider
from scipy.special import logsumexp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'figure.dpi': 100})
rng = np.random.default_rng(42)

# ── 2-D toy dataset: mixture of 3 Gaussians on the vertices of a triangle ──
MEANS = np.array([[0.0, 1.5], [-1.3, -0.75], [1.3, -0.75]])
SIGMA0 = 0.28          # within-cluster std

def sample_data(n, seed=0):
    rng_ = np.random.default_rng(seed)
    k = rng_.integers(0, 3, n)
    return MEANS[k] + rng_.standard_normal((n, 2)) * SIGMA0

X0 = sample_data(800)
print(f"Dataset: {X0.shape}  (2-D, 3 Gaussian clusters)")"""))

D.append(md(r"""<a id="part1"></a>
# Part 1: The Forward Diffusion Process

## 1.1 Gradually adding noise

The forward process is a fixed (non-learned) Markov chain that *gradually corrupts* data by adding small amounts of Gaussian noise at each of $T$ steps:

$$\boxed{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t\mathbf{I}\right), \qquad t = 1, \ldots, T.}$$

The scalars $0 < \beta_1 < \beta_2 < \cdots < \beta_T < 1$ form the **noise schedule**: small early, larger later. At step $t$, the mean is shrunk by $\sqrt{1-\beta_t}$ and independent Gaussian noise with variance $\beta_t$ is added.

## 1.2 Closed form: $q(\mathbf{x}_t \mid \mathbf{x}_0)$

A key algebraic fact: applying the Markov chain $t$ times gives a Gaussian with a closed form. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

$$\boxed{q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1-\bar{\alpha}_t)\mathbf{I}\right).}$$

Equivalently, via the reparameterization trick with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}.$$

As $t \to T$: $\bar{\alpha}_T \approx 0$, so $\mathbf{x}_T \approx \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$. The data is completely destroyed. The reverse process must learn to undo this."""))

D.append(code(r"""# ── Noise schedule (linear, as in Ho et al. 2020) ───────────────────────────
T = 400
betas      = np.linspace(1e-4, 0.02, T)
alphas     = 1.0 - betas
alpha_bars = np.cumprod(alphas)           # ᾱ_t = ∏_{s=1}^t α_s

def forward_sample(x0, t):
    # x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
    ab = alpha_bars[t]
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps, eps

# ── Visualise the forward process at 5 time-steps ────────────────────────────
ts_show = [0, 40, 120, 250, 399]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))

for ax, t in zip(axes, ts_show):
    Xt, _ = forward_sample(X0, t)
    ax.scatter(Xt[:, 0], Xt[:, 1], c='#3498db', s=6, alpha=0.5, edgecolors='none')
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ab = alpha_bars[t]
    ax.set_title(fr'$t={t}$''\n'fr'$\bar{{\alpha}}_t={ab:.3f}$', fontsize=11)

axes[0].set_title(r'$t=0$  (data)' + '\n' + r'$\bar{\alpha}_0=1.00$', fontsize=11)
axes[-1].set_title(r'$t=399$  (noise)' + '\n' + fr'$\bar{{\alpha}}_T={alpha_bars[-1]:.3f}$',
                   fontsize=11)
fig.suptitle('Forward diffusion: gradual corruption of the 3-cluster dataset', fontsize=12)
plt.tight_layout(); plt.show()"""))

D.append(code(r"""def _plot_fwd_step(t=0):
    Xt, _ = forward_sample(X0, t)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].scatter(X0[:, 0], X0[:, 1], c='#3498db', s=8, alpha=0.5, edgecolors='none')
    axes[0].set_title(r'Original data  $\mathbf{x}_0$')
    axes[1].scatter(Xt[:, 0], Xt[:, 1], c='#e74c3c', s=8, alpha=0.5, edgecolors='none')
    axes[1].set_title(fr'Noisy data  $\mathbf{{x}}_{{t}}$,  '
                      fr'$t={t}$,  $\sqrt{{\bar{{\alpha}}_t}}={np.sqrt(alpha_bars[t]):.3f}$')
    for ax in axes:
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()

interact(_plot_fwd_step,
         t=IntSlider(min=0, max=T-1, step=5, value=0, description='t:'));"""))

D.append(md(r"""<a id="part2"></a>
# Part 2: Score Functions

## 2.1 What is a score?

The **score function** of a distribution $p(\mathbf{x})$ is the gradient of its log-density:
$$\mathbf{s}(\mathbf{x}) := \nabla_{\mathbf{x}} \log p(\mathbf{x}).$$

The score *points in the direction of increasing probability density* — it is a vector field that, in a sense, shows where the data is. If you start from any point and follow the score (Langevin dynamics), you drift toward high-density regions.

**Example — Gaussian:** For $p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \sigma^2\mathbf{I})$:
$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2}.$$
The score points toward the mean, with magnitude inversely proportional to $\sigma^2$.

**Mixture of Gaussians:** For $p(\mathbf{x}) = \sum_k w_k \mathcal{N}(\boldsymbol{\mu}_k, \sigma^2\mathbf{I})$ the score is the *responsibility-weighted average* of per-component scores:
$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \sum_{k} r_k(\mathbf{x})\cdot\left(-\frac{\mathbf{x}-\boldsymbol{\mu}_k}{\sigma^2}\right), \qquad r_k(\mathbf{x}) = \frac{w_k\,\mathcal{N}(\mathbf{x};\boldsymbol{\mu}_k,\sigma^2\mathbf{I})}{\sum_j w_j\,\mathcal{N}(\mathbf{x};\boldsymbol{\mu}_j,\sigma^2\mathbf{I})}.$$

## 2.2 Score of the noisy marginal $q_t$

Because the forward process has a closed form, so does the noisy marginal. Starting from our 3-Gaussian data with per-cluster variance $\sigma_0^2$, the marginal at noise level $t$ is another mixture of Gaussians:

$$q_t(\mathbf{x}) = \frac{1}{3}\sum_{k=1}^{3} \mathcal{N}\!\left(\mathbf{x};\; \sqrt{\bar{\alpha}_t}\,\boldsymbol{\mu}_k,\; \sigma_t^2\,\mathbf{I}\right), \qquad \sigma_t^2 = \bar{\alpha}_t\sigma_0^2 + (1-\bar{\alpha}_t).$$

We can therefore compute the **true score** $\nabla_\mathbf{x} \log q_t(\mathbf{x})$ analytically — no network needed. This is what makes our toy dataset pedagogically ideal."""))

D.append(code(r"""def log_gauss(x, mu, var):
    # log N(x; mu, var*I),  x: (N,2), mu: (2,)
    diff = x - mu
    return -0.5 * np.sum(diff**2, axis=1) / var - np.log(2 * np.pi * var)

def score_qt(x, t):
    # Analytical score of q_t for our 3-Gaussian dataset
    ab   = alpha_bars[t]
    var  = ab * SIGMA0**2 + (1.0 - ab)          # σ_t²
    nmu  = np.sqrt(ab) * MEANS                   # scaled cluster centres (3,2)

    log_p = np.stack([log_gauss(x, nmu[k], var) for k in range(3)], axis=0)  # (3,N)
    log_r = log_p - logsumexp(log_p, axis=0, keepdims=True)                   # log-responsibilities
    r = np.exp(log_r)                                                          # (3,N)

    score = np.zeros_like(x)
    for k in range(3):
        score += r[k, :, np.newaxis] * (-(x - nmu[k]) / var)
    return score

# ── Visualise score fields at three noise levels ─────────────────────────────
grid_1d = np.linspace(-3.2, 3.2, 22)
gx, gy  = np.meshgrid(grid_1d, grid_1d)
grid    = np.c_[gx.ravel(), gy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
ts_score = [0, 80, 300]
titles   = [r'$t=0$  (clean data score)',
            r'$t=80$  (mild noise)',
            r'$t=300$  (heavy noise)']

for ax, t, title in zip(axes, ts_score, titles):
    Xt, _  = forward_sample(X0, t)
    s      = score_qt(grid, t)
    mag    = np.linalg.norm(s, axis=1).reshape(gx.shape)
    su, sv = s[:, 0].reshape(gx.shape), s[:, 1].reshape(gx.shape)
    # Normalise arrow length for readability; colour by magnitude
    norm   = np.sqrt(su**2 + sv**2 + 1e-8)
    ax.quiver(gx, gy, su/norm, sv/norm, mag, cmap='plasma',
              scale=30, alpha=0.75, width=0.004)
    ax.scatter(Xt[:300, 0], Xt[:300, 1], c='#3498db', s=4, alpha=0.35, edgecolors='none')
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)

fig.suptitle(r'Score field $\nabla_\mathbf{x}\log q_t(\mathbf{x})$'
             '  (arrows = direction, colour = magnitude)', fontsize=12)
plt.tight_layout(); plt.show()"""))

D.append(md(r"""**Observations:**
- At $t=0$ the score field has three sharp "sinks" pointing toward the cluster centres; the field is strong and well-localised.
- As $t$ grows, noise blurs the clusters together. The score field becomes smoother and, for large $t$, converges to $-\mathbf{x}$ (the score of $\mathcal{N}(\mathbf{0},\mathbf{I})$).
- This smooth-to-sharp transition is the key intuition: **at large $t$ the score is easy to estimate** (it is nearly Gaussian); at small $t$ it requires distinguishing fine structure. A learned network $\mathbf{s}_\theta(\mathbf{x}_t, t)$ must handle the full range.

## 2.3 Langevin dynamics: sampling via the score

Given the score, we can sample from $p$ via **annealed Langevin dynamics**:
$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \frac{\eta}{2}\,\nabla_{\mathbf{x}}\log p(\mathbf{x}^{(k)}) + \sqrt{\eta}\,\boldsymbol{\xi}^{(k)}, \qquad \boldsymbol{\xi}^{(k)} \sim \mathcal{N}(\mathbf{0},\mathbf{I}).$$
As $\eta \to 0$ and steps $\to \infty$, the chain converges to $p$. The stochastic term prevents collapse to a mode.

**Experiment:** The code below runs Langevin dynamics with the *analytical* score $\nabla_\mathbf{x}\log q_0$ on our toy distribution."""))

D.append(code(r"""def langevin_sample(score_fn, n_samples=300, n_steps=800, eta=0.02, seed=1):
    rng_ = np.random.default_rng(seed)
    x = rng_.standard_normal((n_samples, 2)) * 2.0   # initialise from wide Gaussian
    snaps = {0: x.copy()}
    for k in range(1, n_steps + 1):
        s = score_fn(x)
        x = x + (eta / 2) * s + np.sqrt(eta) * rng_.standard_normal(x.shape)
        if k in (50, 200, 500, n_steps):
            snaps[k] = x.copy()
    return snaps

snaps = langevin_sample(lambda x: score_qt(x, 0))

fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
for ax, (step, Xs) in zip(axes, snaps.items()):
    ax.scatter(Xs[:, 0], Xs[:, 1], c='#9b59b6', s=7, alpha=0.55, edgecolors='none')
    # Overlay true cluster centres
    ax.scatter(MEANS[:, 0], MEANS[:, 1], c='gold', s=80, edgecolors='k', lw=0.8,
               zorder=5, marker='*')
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'step {step}', fontsize=11)

fig.suptitle(r'Langevin dynamics with analytical score $\nabla_\mathbf{x}\log q_0$'
             '  (★ = cluster centres)', fontsize=12)
plt.tight_layout(); plt.show()"""))

D.append(md(r"""<a id="part3"></a>
# Part 3: The Reverse Process and DDPM

## 3.1 Reversing the forward chain

Starting from noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0},\mathbf{I})$, we want to run the Markov chain *backwards* to recover a sample from the data. The reverse transition is:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\; \tilde{\beta}_t\,\mathbf{I}\right),$$
$$\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t, \qquad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t.$$

The problem: this requires knowing $\mathbf{x}_0$, which is exactly what we are trying to generate!

## 3.2 DDPM: predict the noise, reconstruct $\mathbf{x}_0$

Ho et al. (2020) train a network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ to **predict the noise** $\boldsymbol{\epsilon}$ that was added at step $t$. Given the prediction, we estimate:
$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}},$$
and substitute into $\tilde{\boldsymbol{\mu}}_t$. This simplifies to:

$$\boxed{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right).}$$

**Connection to score:** The predicted noise and the score are related by
$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \approx -\sqrt{1-\bar{\alpha}_t}\;\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t).$$
Predicting noise is equivalent to estimating the score — DDPM and score-based models are two views of the same idea.

## 3.3 Sampling algorithm

Starting from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0},\mathbf{I})$, iterate for $t = T, T-1, \ldots, 1$:
$$\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) + \sqrt{\tilde{\beta}_t}\,\mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0},\mathbf{I}) \text{ if } t > 1 \text{, else } \mathbf{z}=\mathbf{0}.$$"""))

D.append(code(r"""def ddpm_reverse(n_samples=300, seed=2):
    # Use the analytical score in place of a trained network:
    # eps_hat(x_t, t) = -sqrt(1 - abar_t) * score_qt(x_t, t)
    rng_ = np.random.default_rng(seed)
    x = rng_.standard_normal((n_samples, 2))   # x_T ~ N(0,I)
    snaps = {T: x.copy()}

    for t in range(T - 1, -1, -1):
        ab_t   = alpha_bars[t]
        s_t    = score_qt(x, t)
        eps_hat = -np.sqrt(1.0 - ab_t) * s_t    # analytical noise prediction

        mu = (x - betas[t] / np.sqrt(1.0 - ab_t) * eps_hat) / np.sqrt(alphas[t])

        if t > 0:
            ab_prev   = alpha_bars[t - 1]
            beta_tilde = betas[t] * (1.0 - ab_prev) / (1.0 - ab_t)
            x = mu + np.sqrt(beta_tilde) * rng_.standard_normal(x.shape)
        else:
            x = mu

        if t in (300, 200, 100, 40, 0):
            snaps[t] = x.copy()

    return snaps

rev_snaps = ddpm_reverse()
steps_show = [T, 300, 200, 100, 40, 0]
labels     = [f't={s}' for s in steps_show]
labels[-1] = 't=0  (generated)'

fig, axes = plt.subplots(1, 6, figsize=(16, 3.0))
for ax, t, lbl in zip(axes, steps_show, labels):
    Xs = rev_snaps[t]
    ax.scatter(Xs[:, 0], Xs[:, 1], c='#2ecc71', s=7, alpha=0.55, edgecolors='none')
    ax.scatter(MEANS[:, 0], MEANS[:, 1], c='gold', s=60, edgecolors='k', lw=0.7,
               zorder=5, marker='*')
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lbl, fontsize=10)

fig.suptitle('DDPM reverse diffusion  (★ = true cluster centres)', fontsize=12)
plt.tight_layout(); plt.show()"""))

D.append(code(r"""def _plot_reverse_step(t=T):
    if t not in rev_snaps:
        # find nearest available snapshot
        available = sorted(rev_snaps.keys())
        t = min(available, key=lambda s: abs(s - t))
    Xs = rev_snaps[t]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(Xs[:, 0], Xs[:, 1], c='#2ecc71', s=10, alpha=0.6, edgecolors='none')
    ax.scatter(MEANS[:, 0], MEANS[:, 1], c='gold', s=100, edgecolors='k',
               lw=0.8, zorder=5, marker='*')
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'Reverse diffusion at t = {t}', fontsize=13)
    plt.tight_layout(); plt.show()

available_t = sorted(rev_snaps.keys(), reverse=True)
interact(_plot_reverse_step,
         t=IntSlider(min=0, max=T, step=1, value=T, description='t:'));"""))

D.append(md(r"""<a id="part4"></a>
# Part 4: Training — Denoising Score Matching

## 4.1 The training objective

In practice $\boldsymbol{\epsilon}_\theta$ is a neural network (a U-Net for images). Training minimises:

$$\boxed{\mathcal{L}_{\mathrm{DDPM}} = \mathbb{E}_{t,\,\mathbf{x}_0 \sim q_0,\,\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})}\!\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\; t\right)\right\|^2\right].}$$

**Procedure** (one training step):
1. Sample $\mathbf{x}_0 \sim q_0$ (a training example), $t \sim \mathrm{Uniform}\{1, \ldots, T\}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$.
2. Compute $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$ (one forward step, closed form).
3. Predict $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$.
4. Gradient step on $\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}\|^2$.

This is a **denoising** objective — remarkably similar to the denoising autoencoders studied long before diffusion models. The insight of score matching is that this objective, when summed over all noise levels, is equivalent to matching $\boldsymbol{\epsilon}_\theta$ to the true score field.

## 4.2 Why denoising = score matching

The conditional score of the forward kernel satisfies:
$$\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0}{1-\bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}.$$
Minimising the denoising loss is equivalent to minimising $\mathbb{E}\big[\big\|\mathbf{s}_\theta(\mathbf{x}_t,t) - \nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)\big\|^2\big]$ (the true score matching objective), a result due to Vincent (2011).

> **Exercise 4.1.** *(Deriving $q(\mathbf{x}_t|\mathbf{x}_0)$)*
>
> **(a)** Prove the closed form $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$ by induction on $t$. Use the fact that the sum of two independent Gaussians is Gaussian, and that $\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}$.
>
> **(b)** Show that as $t \to \infty$ with a linear schedule $\beta_t \in (0,1)$, we have $\bar\alpha_t \to 0$ and hence $q(\mathbf{x}_t|\mathbf{x}_0) \to \mathcal{N}(\mathbf{0},\mathbf{I})$.
>
> **(c)** Verify the identity $\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t|\mathbf{x}_0) = -\boldsymbol{\epsilon}/\sqrt{1-\bar\alpha_t}$ and confirm that this gives the equivalence between denoising and score matching."""))

D.append(code(r"""# Demonstrate the denoising objective on one random training step
rng2 = np.random.default_rng(99)
x0_batch = sample_data(128, seed=3)

losses = []
ts_all = rng2.integers(0, T, size=128)
for i in range(128):
    t_i = int(ts_all[i])
    xt, eps = forward_sample(x0_batch[[i]], t_i)
    # "perfect" denoiser: eps_hat = -sqrt(1-ab)*score = eps  (on expectation)
    eps_hat = -np.sqrt(1 - alpha_bars[t_i]) * score_qt(xt, t_i)
    losses.append(float(np.mean((eps - eps_hat)**2)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(ts_all, bins=30, color='#3498db', edgecolor='k', lw=0.4)
axes[0].set_xlabel('Sampled time-step $t$')
axes[0].set_ylabel('Count')
axes[0].set_title('Uniform sampling of $t$ during training')

axes[1].scatter(ts_all, losses, s=10, alpha=0.5, color='#e74c3c', edgecolors='none')
axes[1].set_xlabel('Time-step $t$')
axes[1].set_ylabel(r'$\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}\|^2$')
axes[1].set_title('Denoising loss per sample (analytical predictor)')

plt.tight_layout(); plt.show()
print("Mean denoising loss:", np.mean(losses).round(4))"""))

D.append(md(r"""<a id="part5"></a>
# Part 5: Extensions and the Broader Picture

## 5.1 DDIM — deterministic sampling

The standard DDPM sampler adds fresh noise at each reverse step, requiring $T \approx 1000$ steps for high quality. **DDIM** (Song et al., 2021) finds a *non-Markovian* process that shares the same marginals $q_t$ but admits a deterministic reverse ODE:
$$\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\frac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}_\theta}{\sqrt{\bar\alpha_t}}}_{\text{predicted }\mathbf{x}_0} + \sqrt{1-\bar\alpha_{t-1}}\,\boldsymbol{\epsilon}_\theta.$$
This allows generation in as few as 20–50 steps, accelerating diffusion by 50×.

## 5.2 Continuous time: score SDEs

Song et al. (2021) unify all of these variants. The forward process is modelled as an SDE:
$$d\mathbf{x} = \mathbf{f}(\mathbf{x},t)\,dt + g(t)\,d\mathbf{w},$$
and the reverse-time SDE (Anderson, 1982) is:
$$d\mathbf{x} = \bigl[\mathbf{f}(\mathbf{x},t) - g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\bigr]\,dt + g(t)\,d\bar{\mathbf{w}}.$$
The score $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ is exactly what the network $\mathbf{s}_\theta$ learns. Setting the noise term in the reverse SDE to zero gives an **ODE** (the probability-flow ODE), enabling likelihood evaluation and fast sampling.

| Paradigm | Training objective | Sampling | Key paper |
|----------|-------------------|----------|-----------|
| **DDPM** | Denoising $\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta\|^2$ | Stochastic, $T\sim1000$ steps | Ho et al. 2020 |
| **Score matching** | $\|\mathbf{s}_\theta - \nabla\log p_t\|^2$ | Annealed Langevin | Song & Ermon 2019 |
| **DDIM** | Same as DDPM | Deterministic ODE, 20–50 steps | Song et al. 2021 |
| **Score SDE** | Unified via SDE framework | SDE or ODE | Song et al. 2021 |

> **Exercise 5.1.** *(Score of a Gaussian)*
>
> **(a)** Verify that for $p(\mathbf{x}) = \mathcal{N}(\boldsymbol\mu, \sigma^2\mathbf{I})$, the score is $\nabla_\mathbf{x}\log p(\mathbf{x}) = -(\mathbf{x}-\boldsymbol\mu)/\sigma^2$.
>
> **(b)** What is $\lim_{t\to T}\nabla_\mathbf{x}\log q_t(\mathbf{x})$? Confirm your answer using `score_qt(x, T-1)` on a grid of points.
>
> **(c)** *(Challenge)* Implement a simplified DDIM sampler on our 2-D toy distribution, using fewer steps (say 40 instead of 400). Compare the quality of generated samples to DDPM."""))

D.append(md(r"""---
# Summary

| Concept | Key formula / takeaway |
|---------|------------------------|
| **Forward process** | $q(\mathbf{x}_t\|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0,\,(1{-}\bar\alpha_t)\mathbf{I})$ |
| **Reparameterisation** | $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1{-}\bar\alpha_t}\boldsymbol\epsilon$ |
| **Score function** | $\mathbf{s}(\mathbf{x}) = \nabla_\mathbf{x}\log p(\mathbf{x})$; points toward high-density regions |
| **DDPM objective** | $\mathbb{E}\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t)\|^2$ — denoising equals score matching |
| **Reverse step** | $\boldsymbol\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1{-}\bar\alpha_t}}\boldsymbol\epsilon_\theta)$ |
| **Continuous limit** | Forward SDE + reverse SDE (via score) = score-based generative model |"""))

# ════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — reinforcement_learning.ipynb
# ════════════════════════════════════════════════════════════════════════════

R = []

R.append(md(r"""# Reinforcement Learning: From Bellman to Policy Gradients to RLHF
## Reward-driven learning and the mathematics of sequential decision making

**MAT 4953/6973 — Mathematical Foundations of AI** (Spring 2026, UTSA)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/reinforcement_learning.ipynb)

---"""))

R.append(md(r"""> **Prerequisites:** Probability (expectations, conditional distributions), calculus (gradients), familiarity with neural networks. No RL background assumed.

The supervised and unsupervised paradigms we have studied so far share a common structure: training data is given, and we minimise a loss derived from that data. **Reinforcement learning** (RL) is fundamentally different: an *agent* interacts with an *environment*, receives *rewards* as feedback, and must learn a *policy* — a mapping from situations to actions — that maximises long-run cumulative reward. There are no labelled examples; the agent must discover good behaviour through trial and error.

RL is the backbone of many landmark achievements (AlphaGo, game-playing AIs, robotic control) and, critically for this course, the fine-tuning stage that aligns large language models to human preferences (**RLHF**).

**Outline**
1. [The RL framework: MDPs](#part1)
2. [Value functions and the Bellman equation](#part2)
3. [Policy gradient methods](#part3)
4. [Deep RL and function approximation](#part4)
5. [RLHF: aligning language models](#part5)"""))

R.append(code(r"""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from ipywidgets import interact, IntSlider, FloatSlider

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'figure.dpi': 100})
rng = np.random.default_rng(0)
print("Libraries loaded.")"""))

R.append(md(r"""<a id="part1"></a>
# Part 1: The RL Framework — Markov Decision Processes

## 1.1 MDPs

A **Markov Decision Process** (MDP) is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

| Symbol | Name | Description |
|--------|------|-------------|
| $\mathcal{S}$ | State space | Set of all possible situations the agent can be in |
| $\mathcal{A}$ | Action space | Set of all decisions the agent can make |
| $P(s' \mid s, a)$ | Transition dynamics | Probability of reaching $s'$ from $s$ after action $a$ |
| $R(s, a)$ | Reward function | Scalar signal received after taking action $a$ in state $s$ |
| $\gamma \in [0,1)$ | Discount factor | How much to value future rewards vs. immediate ones |

The agent's goal is to find a **policy** $\pi(a \mid s)$ — a distribution over actions given the current state — that maximises the **expected discounted return**:
$$J(\pi) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right].$$

The **Markov property** is essential: the future depends only on the current state, not the full history. This property is what makes the Bellman equation tractable.

## 1.2 A gridworld MDP

We use a $5\times5$ gridworld as a concrete running example throughout this notebook."""))

R.append(code(r"""# ── Gridworld definition ─────────────────────────────────────────────────────
SIZE   = 5
N_S    = SIZE * SIZE                 # 25 states
N_A    = 4                           # up=0, down=1, left=2, right=3
GOAL   = N_S - 1                     # state 24 = cell (4,4)
WALLS  = {7, 11, 17}                 # blocked cells
DELTA  = [(-1,0),(1,0),(0,-1),(0,1)] # (dr, dc) for each action

def step(s, a):
    # Returns (s_next, reward)
    if s == GOAL: return s, 0.0
    r, c = divmod(s, SIZE)
    dr, dc = DELTA[a]
    rn, cn = max(0, min(SIZE-1, r+dr)), max(0, min(SIZE-1, c+dc))
    sn = rn * SIZE + cn
    if sn in WALLS: sn = s           # bounce off wall
    reward = 10.0 if sn == GOAL else -0.1
    return sn, reward

def draw_grid(ax, V=None, policy=None, title=''):
    ax.set_xlim(0, SIZE); ax.set_ylim(0, SIZE)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=11)
    cmap = plt.cm.YlGn
    norm = Normalize(vmin=0, vmax=max(1.0, V.max()) if V is not None else 1.0)

    for s in range(N_S):
        r, c = divmod(s, SIZE)
        y, x = SIZE - 1 - r, c       # flip y for display
        if s in WALLS:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='#555555'))
            continue
        fc = cmap(norm(V[s])) if V is not None else '#f0f0f0'
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=fc,
                                    edgecolor='#aaaaaa', lw=0.8))
        if s == GOAL:
            ax.text(x+0.5, y+0.5, 'G', ha='center', va='center',
                    fontsize=13, fontweight='bold', color='#27ae60')
        elif s == 0:
            ax.text(x+0.5, y+0.5, 'S', ha='center', va='center',
                    fontsize=11, color='#2980b9')
        elif V is not None:
            ax.text(x+0.5, y+0.5, f'{V[s]:.1f}', ha='center', va='center',
                    fontsize=8, color='#333333')
        if policy is not None and s not in WALLS and s != GOAL:
            arrows = ['^','v','<','>']
            ax.text(x+0.5, y+0.5, arrows[policy[s]], ha='center', va='center',
                    fontsize=14, color='#c0392b')

fig, ax = plt.subplots(figsize=(4, 4))
draw_grid(ax, title='Gridworld  (S=start, G=goal, ■=wall)')
plt.tight_layout(); plt.show()"""))

R.append(md(r"""<a id="part2"></a>
# Part 2: Value Functions and the Bellman Equation

## 2.1 State-value function

The **state-value function** $V^\pi : \mathcal{S} \to \mathbb{R}$ measures how good it is to be in state $s$ when following policy $\pi$:
$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t R(s_t,a_t) \;\Big|\; s_0 = s\right].$$

The **Bellman equation** expresses $V^\pi$ as a fixed-point equation — the value of a state equals the expected immediate reward plus the discounted value of the next state:

$$\boxed{V^\pi(s) = \sum_{a} \pi(a\mid s)\left[R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)\,V^\pi(s')\right].}$$

## 2.2 Optimal value and Bellman optimality

The **optimal value function** $V^*$ is achieved by the best possible policy:
$$\boxed{V^*(s) = \max_{a}\left[R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)\,V^*(s')\right].}$$

This is the **Bellman optimality equation**. It is a fixed-point equation: $V^* = \mathcal{T} V^*$ where $\mathcal{T}$ is the Bellman optimality operator. Because $\mathcal{T}$ is a $\gamma$-contraction (in the $\ell^\infty$ norm), repeated application of $\mathcal{T}$ converges to the unique fixed point $V^*$.

**Value iteration** exploits this:

$$V_{k+1}(s) \leftarrow \max_{a}\left[R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)\, V_k(s')\right], \qquad k = 0, 1, 2, \ldots$$"""))

R.append(code(r"""def value_iteration(gamma=0.95, n_iter=60, eps=1e-6):
    V = np.zeros(N_S)
    policy = np.zeros(N_S, dtype=int)
    history = [V.copy()]

    for it in range(n_iter):
        V_new = V.copy()
        for s in range(N_S):
            if s == GOAL or s in WALLS:
                continue
            q_vals = []
            for a in range(N_A):
                sn, r = step(s, a)
                q_vals.append(r + gamma * V[sn])
            V_new[s]    = max(q_vals)
            policy[s]   = int(np.argmax(q_vals))
        history.append(V_new.copy())
        if np.max(np.abs(V_new - V)) < eps:
            print(f"Converged at iteration {it+1}")
            break
        V = V_new

    return V, policy, history

V_star, pi_star, V_history = value_iteration()

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
iters_show = [0, 5, 20, len(V_history)-1]
for ax, it in zip(axes, iters_show):
    draw_grid(ax, V=V_history[it],
              policy=pi_star if it == len(V_history)-1 else None,
              title=f'Iteration {it}')
fig.suptitle(r'Value iteration: $V_k \to V^*$  (colour = value, arrows = optimal policy)',
             fontsize=12)
plt.tight_layout(); plt.show()"""))

R.append(code(r"""def _plot_vi(iteration=0):
    it = min(iteration, len(V_history)-1)
    pol = pi_star if it == len(V_history)-1 else None
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    draw_grid(ax, V=V_history[it], policy=pol,
              title=f'Value iteration — step {it}')
    plt.tight_layout(); plt.show()

interact(_plot_vi, iteration=IntSlider(min=0, max=len(V_history)-1,
                                        step=1, value=0, description='Iter:'));"""))

R.append(md(r"""> **Exercise 2.1.** *(Bellman fixed point)*
>
> **(a)** Prove that the Bellman optimality operator $\mathcal{T}$, defined by $(\mathcal{T}V)(s) = \max_a[R(s,a) + \gamma\sum_{s'}P(s'|s,a)V(s')]$, is a $\gamma$-contraction in the $\ell^\infty$ norm: $\|\mathcal{T}V - \mathcal{T}W\|_\infty \le \gamma\|V-W\|_\infty$.
>
> **(b)** Use the contraction property to bound the error after $k$ iterations of value iteration: $\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty$. How many iterations are needed to reach error $\varepsilon = 0.01$ for $\gamma = 0.95$?
>
> **(c)** Modify the gridworld to add a second goal state with reward $+5$ at position $(0,4)$. Re-run value iteration. Does the optimal policy change?"""))

R.append(md(r"""<a id="part3"></a>
# Part 3: Policy Gradient Methods

## 3.1 Why not always use value iteration?

Value iteration requires knowing $P(s' \mid s, a)$ explicitly (a *model* of the environment) and storing a value for every state. For large or continuous state spaces (e.g., raw pixel inputs or physical sensor readings), both are infeasible.

**Policy gradient** methods directly parametrise and optimise the policy $\pi_\theta(a \mid s)$ — a neural network — without needing a model. They optimise $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$ by gradient ascent.

## 3.2 The Policy Gradient Theorem

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\cdot G_t\right],}$$

where $G_t = \sum_{k=t}^{T}\gamma^{k-t}R(s_k,a_k)$ is the return from step $t$ onwards.

**Intuition:** $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is the direction to increase the probability of action $a_t$; $G_t$ weights this by how good the subsequent trajectory was. Actions followed by high returns are reinforced; actions followed by low returns are suppressed.

The REINFORCE algorithm (Williams, 1992) uses Monte Carlo sampling to estimate this expectation:

```
Repeat:
    Collect episode τ = (s₀,a₀,r₀, s₁,a₁,r₁, ...) by rolling out π_θ
    For each t: compute Gₜ = Σ_{k≥t} γ^{k−t} rₖ
    Update θ ← θ + η · Σ_t ∇_θ log π_θ(aₜ|sₜ) · Gₜ
```

**Variance reduction:** $G_t$ has high variance. A common fix is to subtract a *baseline* $b(s_t)$ (often $V^\pi(s_t)$), giving the *advantage* $A_t = G_t - b(s_t)$. This is the actor-critic idea."""))

R.append(code(r"""# ── Simple REINFORCE on the gridworld ────────────────────────────────────────
# Policy: softmax over a linear function of a one-hot state encoding
# theta: (N_A, N_S) weight matrix

def policy_probs(theta, s):
    logits = theta[:, s]
    logits = logits - logits.max()   # numerical stability
    probs  = np.exp(logits); probs /= probs.sum()
    return probs

def collect_episode(theta, gamma=0.95, max_steps=200, rng_=None):
    if rng_ is None: rng_ = np.random.default_rng()
    s = 0; traj = []
    for _ in range(max_steps):
        probs = policy_probs(theta, s)
        a     = int(rng_.choice(N_A, p=probs))
        sn, r = step(s, a)
        traj.append((s, a, r))
        s = sn
        if s == GOAL: break
    return traj

def reinforce(n_episodes=3000, lr=0.02, gamma=0.95, seed=7):
    rng_ = np.random.default_rng(seed)
    theta = rng_.standard_normal((N_A, N_S)) * 0.01
    returns_log = []

    for ep in range(n_episodes):
        traj = collect_episode(theta, gamma, rng_=rng_)
        T_ep = len(traj)

        # Compute returns
        Gs = np.zeros(T_ep)
        G  = 0.0
        for t in reversed(range(T_ep)):
            G = traj[t][2] + gamma * G
            Gs[t] = G

        # Baseline: mean return (reduces variance)
        Gs = Gs - Gs.mean()

        # Gradient update
        for t, (s, a, _) in enumerate(traj):
            probs = policy_probs(theta, s)
            grad  = -probs.copy()
            grad[a] += 1.0                # ∇ log π(a|s) for softmax
            theta[:, s] += lr * grad * Gs[t]

        returns_log.append(sum(r for _, _, r in traj))

    return theta, returns_log

print("Training REINFORCE on gridworld …")
theta_rl, ret_log = reinforce(n_episodes=3000)
print("Done.")

# Smooth returns
window = 100
smoothed = np.convolve(ret_log, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ret_log, alpha=0.3, color='#3498db', lw=0.8)
axes[0].plot(np.arange(window-1, len(ret_log)), smoothed, color='#3498db', lw=2)
axes[0].axhline(0, color='k', lw=0.5, ls='--')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Episode return')
axes[0].set_title('REINFORCE learning curve')

V_rl = np.array([np.dot(policy_probs(theta_rl, s),
                         [step(s,a)[1] + 0.95 * V_star[step(s,a)[0]] for a in range(N_A)])
                  for s in range(N_S)])
draw_grid(axes[1], V=V_rl,
          policy=np.array([np.argmax([step(s,a)[1] + 0.95*V_star[step(s,a)[0]]
                                       for a in range(N_A)]) for s in range(N_S)]),
          title='Learned policy (REINFORCE)')
plt.tight_layout(); plt.show()"""))

R.append(md(r"""<a id="part4"></a>
# Part 4: Deep RL — Scaling Up with Neural Networks

The key step from classical RL to **deep RL** is replacing the tabular policy (a table with one entry per state) with a neural network $\pi_\theta$ or value function $V_\theta$. This allows generalisation across similar states and makes RL applicable to high-dimensional inputs like images.

**DQN** (Mnih et al., 2015): a deep network $Q_\theta(s, a)$ approximates the action-value function $Q^*(s,a) = R(s,a) + \gamma\max_{a'}Q^*(s',a')$. The network is trained by minimising the Bellman residual:
$$\mathcal{L}(\theta) = \mathbb{E}\!\left[\left(R + \gamma\max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right],$$
where $\theta^-$ are *target network* parameters updated slowly. Two tricks stabilise training:
- **Experience replay**: store transitions $(s,a,r,s')$ in a buffer; sample random mini-batches.
- **Target network**: avoid moving target by delaying updates to $\theta^-$.

**Proximal Policy Optimisation (PPO)** (Schulman et al., 2017) is now the dominant algorithm for both continuous control and RLHF. It is an actor-critic method that clips the policy update:
$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}\!\left[\min\!\left(r_t(\theta)\,A_t,\;\mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\,A_t\right)\right],$$
where $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$ is the importance ratio and $A_t$ is the advantage. The clip prevents large destabilising policy updates — the "proximal" in PPO."""))

R.append(md(r"""<a id="part5"></a>
# Part 5: RLHF — Reinforcement Learning from Human Feedback

## 5.1 The alignment problem

Large language models pre-trained on text are good at predicting tokens, but not necessarily at producing *helpful*, *harmless*, or *honest* responses. Supervised fine-tuning on curated examples helps but has limits — it is hard to specify every desirable behaviour. RLHF (Christiano et al., 2017; Ouyang et al., 2022) uses RL with a *learned reward model* trained on human preference judgements.

## 5.2 Three-stage pipeline

**Stage 1 — Supervised fine-tuning (SFT):** Fine-tune the base LLM on high-quality demonstrations to get $\pi_{\text{SFT}}$.

**Stage 2 — Reward model training:** Collect pairs of responses $(y_1, y_2)$ to the same prompt $x$ and ask human annotators: *"Which is better?"* Fit a scalar reward model $R_\phi(x, y)$ using the **Bradley–Terry** preference model:

$$P(y_1 \succ y_2 \mid x) = \sigma\!\left(R_\phi(x,y_1) - R_\phi(x,y_2)\right),$$
$$\mathcal{L}(\phi) = -\mathbb{E}_{(x,y_1,y_2)}\!\left[\log\sigma\!\left(R_\phi(x,y_1) - R_\phi(x,y_2)\right)\right].$$

**Stage 3 — RL fine-tuning:** Optimise the LLM policy $\pi_\theta$ (initialised from $\pi_{\text{SFT}}$) to maximise expected reward while staying close to the SFT baseline:

$$\boxed{\max_\theta \;\mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot|x)}\!\left[R_\phi(x,y)\right] - \lambda\, D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot|x)\;\|\;\pi_{\mathrm{SFT}}(\cdot|x)\right).}$$

The **KL penalty** prevents the policy from "reward-hacking" — generating text that gets a high reward score but deviates so far from the SFT distribution that it becomes incoherent."""))

R.append(code(r"""# ── Visualise the reward–KL tradeoff ────────────────────────────────────────
# Synthetic illustration: reward increases then saturates; KL penalty grows

lam_values = [0.0, 0.1, 0.5, 2.0]
kl_range   = np.linspace(0, 10, 300)

def reward_fn(kl):
    # Reward saturates after some KL divergence (reward hacking regime)
    return 3.0 * (1 - np.exp(-0.8 * kl)) - 0.3 * np.maximum(0, kl - 4)**2

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
reward_curve = reward_fn(kl_range)
axes[0].plot(kl_range, reward_curve, color='#3498db', lw=2.5, label='Reward $R_\\phi$')
axes[0].set_xlabel(r'$D_\mathrm{KL}(\pi_\theta \| \pi_\mathrm{SFT})$')
axes[0].set_ylabel('Value')
axes[0].set_title('Reward model output vs. KL from SFT')
axes[0].legend(fontsize=11)

colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
for lam, c in zip(lam_values, colors):
    obj = reward_fn(kl_range) - lam * kl_range
    opt_idx = np.argmax(obj)
    axes[1].plot(kl_range, obj, color=c, lw=2, label=fr'$\lambda={lam}$')
    axes[1].axvline(kl_range[opt_idx], color=c, lw=0.8, ls='--', alpha=0.7)

axes[1].set_xlabel(r'$D_\mathrm{KL}(\pi_\theta \| \pi_\mathrm{SFT})$')
axes[1].set_ylabel(r'$\mathbb{E}[R_\phi] - \lambda\,D_\mathrm{KL}$')
axes[1].set_title(r'RLHF objective for different $\lambda$ values')
axes[1].legend(fontsize=10)

for ax in axes:
    ax.axhline(0, color='k', lw=0.4, ls='--')
    ax.grid(True, alpha=0.2)

plt.tight_layout(); plt.show()
print("Optimal KL for each λ:")
for lam, c in zip(lam_values, colors):
    obj = reward_fn(kl_range) - lam * kl_range
    print(f"  λ={lam:.1f}  →  KL* ≈ {kl_range[np.argmax(obj)]:.2f}")"""))

R.append(code(r"""# ── Bradley–Terry reward model: likelihood landscape ─────────────────────────
# Illustrate: fitting R(x,y) from pairwise preferences

delta_r = np.linspace(-4, 4, 300)   # R(y1) - R(y2)
prob_y1_wins = 1 / (1 + np.exp(-delta_r))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(delta_r, prob_y1_wins, color='#3498db', lw=2.5)
axes[0].axhline(0.5, color='k', lw=0.6, ls='--')
axes[0].axvline(0.0, color='k', lw=0.6, ls='--')
axes[0].set_xlabel(r'$R_\phi(x,y_1) - R_\phi(x,y_2)$')
axes[0].set_ylabel(r'$P(y_1 \succ y_2)$')
axes[0].set_title('Bradley–Terry preference probability')
axes[0].fill_between(delta_r, 0.5, prob_y1_wins,
                      where=delta_r > 0, alpha=0.15, color='#2ecc71',
                      label=r'$y_1$ preferred')
axes[0].fill_between(delta_r, prob_y1_wins, 0.5,
                      where=delta_r < 0, alpha=0.15, color='#e74c3c',
                      label=r'$y_2$ preferred')
axes[0].legend(fontsize=10)

# Negative log-likelihood
nll = -np.log(prob_y1_wins + 1e-9)
axes[1].plot(delta_r, nll, color='#e74c3c', lw=2.5)
axes[1].set_xlabel(r'$R_\phi(x,y_1) - R_\phi(x,y_2)$')
axes[1].set_ylabel('Negative log-likelihood')
axes[1].set_title(r'Training loss: $-\log\sigma(\Delta R)$')
axes[1].set_ylim(0, 6)
axes[1].grid(True, alpha=0.2)

plt.tight_layout(); plt.show()"""))

R.append(md(r"""## 5.3 Constitutional AI and direct alignment

After RLHF, several variants have been proposed:

- **Direct Preference Optimisation (DPO)** (Rafailov et al., 2023) eliminates the explicit reward model by showing that the RLHF objective has a closed-form solution in terms of preference data, reducing training to supervised learning on comparisons.
- **Constitutional AI** (Anthropic, 2022) replaces human preference labels with a set of principles; the model self-critiques using those principles.

In all cases, the mathematical core is the same: a KL-regularised optimisation of a reward (or preference) signal.

> **Exercise 5.1.** *(Policy gradient derivation)*
>
> **(a)** Prove the policy gradient theorem: $\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta\log\pi_\theta(a|s)\cdot Q^\pi(s,a)]$. Start from $J(\theta) = \sum_s d^\pi(s)\sum_a\pi_\theta(a|s)Q^\pi(s,a)$ where $d^\pi(s)$ is the stationary distribution.
>
> *Hint:* Use $\nabla_\theta\pi_\theta(a|s) = \pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)$ (the log-derivative trick).
>
> **(b)** Show that subtracting a baseline $b(s)$ from $G_t$ does not bias the gradient estimate: $\mathbb{E}_\pi[\nabla_\theta\log\pi_\theta(a|s)\cdot b(s)] = 0$.
>
> **(c)** In the RLHF objective, the KL term can be written as $D_\mathrm{KL}(\pi_\theta\|\pi_\mathrm{SFT}) = \mathbb{E}_{\pi_\theta}[\log\pi_\theta(y|x) - \log\pi_\mathrm{SFT}(y|x)]$. Interpret this: what happens when $\lambda \to 0$? When $\lambda \to \infty$?"""))

R.append(md(r"""---
# Summary

| Concept | Key formula / takeaway |
|---------|------------------------|
| **MDP** | $(\mathcal{S},\mathcal{A},P,R,\gamma)$; goal: maximise $\mathbb{E}[\sum_t\gamma^t R_t]$ |
| **Bellman equation** | $V^\pi(s) = \sum_a\pi(a\|s)[R(s,a)+\gamma\sum_{s'}P(s'\|s,a)V^\pi(s')]$ |
| **Value iteration** | Contraction $\mathcal{T}$ on $V$; converges to $V^*$ at rate $\gamma^k$ |
| **Policy gradient** | $\nabla_\theta J = \mathbb{E}[\nabla_\theta\log\pi_\theta(a\|s)\cdot G_t]$ |
| **RLHF reward model** | $P(y_1\succ y_2) = \sigma(R_\phi(x,y_1)-R_\phi(x,y_2))$ (Bradley–Terry) |
| **RLHF objective** | $\max_\theta\mathbb{E}[R_\phi(x,y)] - \lambda D_\mathrm{KL}(\pi_\theta\|\pi_\mathrm{SFT})$ |"""))

# ════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 — geometric_deep_learning.ipynb
# ════════════════════════════════════════════════════════════════════════════

G = []

G.append(md(r"""# Geometric Deep Learning: Symmetry, Groups, and Graph Neural Networks
## Equivariance as an architectural principle

**MAT 4953/6973 — Mathematical Foundations of AI** (Spring 2026, UTSA)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/geometric_deep_learning.ipynb)

---"""))

G.append(md(r"""> **Prerequisites:** Linear algebra (matrices, linear maps), basic abstract algebra (groups are introduced from scratch), familiarity with feedforward networks and CNNs ([`dnn_architectures_overview.ipynb`](dnn_architectures_overview.ipynb)).

Why do convolutional neural networks work for images? The standard answer — *parameter sharing reduces model size* — is correct but misses the deeper reason. The true answer is that **CNNs encode a symmetry of the problem**: natural images are translation-equivariant — a cat in the top-left corner is the same cat in the bottom-right corner. The architecture *bakes in* this knowledge.

**Geometric deep learning** (Bronstein, Bruna, Cohen, Veličković, 2021) generalises this principle: *build symmetry into the architecture, not into the data*. The same framework unifies CNNs, graph neural networks, recurrent networks, and transformers as special cases of a single blueprint. The mathematical language is **group theory**.

**Outline**
1. [Symmetry and inductive bias](#part1)
2. [Groups, actions, equivariance](#part2)
3. [Graphs and permutation symmetry](#part3)
4. [Graph neural networks and GCNs](#part4)
5. [Expressivity: the Weisfeiler–Lehman test](#part5)
6. [Beyond 2D: 3D equivariance in science](#part6)"""))

G.append(code(r"""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from ipywidgets import interact, IntSlider, Dropdown
from scipy.signal import convolve2d

# networkx for graph drawing
try:
    import networkx as nx
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'networkx', '-q'])
    import networkx as nx

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'figure.dpi': 100})
rng = np.random.default_rng(0)
print(f"NetworkX {nx.__version__} loaded.")"""))

G.append(md(r"""<a id="part1"></a>
# Part 1: Symmetry as an Inductive Bias

## 1.1 Why symmetry matters

Consider training an image classifier. If the network has no built-in knowledge of translation symmetry, it must *learn* from data that a cat at position $(10,10)$ and a cat at position $(50,80)$ are the same object. That requires vastly more labelled examples. **Encoding the symmetry directly reduces the effective size of the hypothesis class**, improving sample efficiency.

The general principle:
- **Invariance:** $f(g \cdot \mathbf{x}) = f(\mathbf{x})$ — the output does not change under symmetry $g$ (e.g., a classifier should give the same label regardless of position).
- **Equivariance:** $f(g \cdot \mathbf{x}) = g \cdot f(\mathbf{x})$ — the output *transforms consistently* with the input (e.g., a feature map should shift when the image shifts).

Invariance and equivariance are related: invariance = equivariance followed by a pooling operation that discards the group action.

## 1.2 Convolution is equivariant to translation

The 2-D convolution $(f * h)[m,n] = \sum_{k,l} f[k,l]\,h[m-k,n-l]$ satisfies:
$$(\mathcal{T}_{(a,b)} f) * h = \mathcal{T}_{(a,b)}(f * h),$$
where $\mathcal{T}_{(a,b)}$ shifts the image by $(a,b)$. This is the **translation-equivariance** of convolution — the defining property of CNNs."""))

G.append(code(r"""# ── Demonstrate translation equivariance of convolution ─────────────────────
rng2 = np.random.default_rng(1)
img = np.zeros((20, 20))
img[6:10, 6:10] = 1.0           # a small square "object"

kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=float)   # Sobel edge detector

def translate(arr, dr, dc):
    result = np.zeros_like(arr)
    r0, r1 = max(0, dr), min(arr.shape[0], arr.shape[0]+dr)
    c0, c1 = max(0, dc), min(arr.shape[1], arr.shape[1]+dc)
    rs0, rs1 = max(0, -dr), min(arr.shape[0], arr.shape[0]-dr)
    cs0, cs1 = max(0, -dc), min(arr.shape[1], arr.shape[1]-dc)
    result[r0:r1, c0:c1] = arr[rs0:rs1, cs0:cs1]
    return result

dr, dc = 4, 3
img_shifted = translate(img, dr, dc)

feat          = convolve2d(img,         kernel, mode='same', boundary='fill')
feat_of_shift = convolve2d(img_shifted, kernel, mode='same', boundary='fill')
shift_of_feat = translate(feat, dr, dc)

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
data = [(img,          'Original $f$'),
        (img_shifted,  r'Shifted $\mathcal{T}f$'),
        (feat,         r'$f * h$  (conv of original)'),
        (feat_of_shift,r'$(\mathcal{T}f) * h$  (conv of shifted)'),
        (shift_of_feat,r'$\mathcal{T}(f * h)$  (shift of conv)'),
        (np.abs(feat_of_shift - shift_of_feat), r'$|(\mathcal{T}f)*h - \mathcal{T}(f*h)|$')]

for ax, (img_, title) in zip(axes.ravel(), data):
    ax.imshow(img_, cmap='RdBu_r', origin='upper',
              vmin=-1.5 if 'conv' in title.lower() else 0,
              vmax=1.5 if 'conv' in title.lower() else 1)
    ax.set_title(title, fontsize=10); ax.axis('off')

fig.suptitle('Convolution is equivariant to translation  '
             r'(bottom-right $\approx 0$ confirms equivariance)', fontsize=12)
plt.tight_layout(); plt.show()
print(f"Max deviation from perfect equivariance: {np.abs(feat_of_shift - shift_of_feat).max():.2e}")"""))

G.append(md(r"""<a id="part2"></a>
# Part 2: Groups, Actions, and Equivariance

## 2.1 Groups

A **group** is a set $G$ with a binary operation $\cdot$ satisfying:
1. *Closure:* $g \cdot h \in G$ for all $g, h \in G$.
2. *Associativity:* $(g \cdot h) \cdot k = g \cdot (h \cdot k)$.
3. *Identity:* $\exists\, e \in G$ such that $e \cdot g = g \cdot e = g$.
4. *Inverses:* $\forall\, g \in G$, $\exists\, g^{-1}$ such that $g \cdot g^{-1} = e$.

**Examples relevant to deep learning:**

| Group | Elements | Operation | Application |
|-------|---------|-----------|-------------|
| $(\mathbb{Z}^2, +)$ | 2-D integer vectors | Vector addition | Translation of image pixels |
| $SO(2)$ | $2\times2$ rotation matrices | Matrix multiplication | Rotation of 2-D images |
| $SO(3)$ | $3\times3$ rotation matrices | Matrix multiplication | Rotation of 3-D molecules |
| $E(3)$ | Rotations + translations in $\mathbb{R}^3$ | Composition | 3-D point clouds, proteins |
| $S_n$ | Permutations of $\{1,\ldots,n\}$ | Function composition | Sets, graphs (relabelling nodes) |

## 2.2 Group actions and equivariance

A **group action** of $G$ on a space $\mathcal{X}$ is a map $\rho: G \times \mathcal{X} \to \mathcal{X}$ (written $g \cdot \mathbf{x}$) satisfying $e \cdot \mathbf{x} = \mathbf{x}$ and $(g h) \cdot \mathbf{x} = g \cdot (h \cdot \mathbf{x})$.

A function $f: \mathcal{X} \to \mathcal{Y}$ is **$G$-equivariant** if:
$$\boxed{f(g \cdot \mathbf{x}) = g \cdot f(\mathbf{x}) \qquad \forall\, g \in G, \; \mathbf{x} \in \mathcal{X}.}$$

**Key theorem:** The space of $G$-equivariant linear maps $f: \mathbb{R}^n \to \mathbb{R}^n$ (where $G$ acts by permutation matrices) is exactly the space of matrices of the form $\alpha I + \beta \mathbf{1}\mathbf{1}^\top$ — *i.e.*, linear combinations of the identity and the all-ones matrix. Any layer of a permutation-equivariant network on sets must have this form!"""))

G.append(code(r"""# ── Visualise: rotation is NOT a symmetry of a standard MLP ─────────────────
# A standard learned filter is not equivariant to rotation.
# We show this by applying a 45° rotation before vs. after a non-symmetric filter.

theta_rot = np.pi / 4    # 45 degrees

def rotate_img(arr, angle):
    from scipy.ndimage import rotate as sci_rotate
    return sci_rotate(arr, np.degrees(angle), reshape=False, order=1)

def apply_filter(arr, filt):
    return convolve2d(arr, filt, mode='same', boundary='fill')

# Asymmetric filter (not rotation-equivariant)
asym_filt = np.array([[2, 1, 0],
                       [1, 0,-1],
                       [0,-1,-2]], dtype=float) / 4.0

# Symmetric filter (rotation-equivariant for 90° rotations)
sym_filt = np.array([[1, 1, 1],
                      [1, 8, 1],
                      [1, 1, 1]], dtype=float) / 16.0

img_blob = np.zeros((24, 24))
img_blob[8:16, 4:10] = 1.0      # asymmetric object

img_rot  = rotate_img(img_blob, theta_rot)

fig, axes = plt.subplots(2, 4, figsize=(13, 6))
row_labels = ['Asymmetric filter\n(NOT equivariant)', 'Isotropic filter\n(equivariant to rotations)']
for row, filt in enumerate([asym_filt, sym_filt]):
    f_then_rot = rotate_img(apply_filter(img_blob, filt), theta_rot)
    rot_then_f = apply_filter(img_rot, filt)
    diff        = np.abs(f_then_rot - rot_then_f)

    for col, (im, ttl) in enumerate([
            (img_blob,    'Input $f$'),
            (img_rot,     r'Rotated input $\rho(g)f$'),
            (f_then_rot,  r'Filter $\to$ rotate: $\rho(g)(Wf)$'),
            (rot_then_f,  r'Rotate $\to$ filter: $W(\rho(g)f)$'),
    ]):
        axes[row][col].imshow(im, cmap='RdBu_r', origin='upper')
        axes[row][col].set_title(ttl, fontsize=9)
        axes[row][col].axis('off')

for row, lbl in enumerate(row_labels):
    axes[row][0].set_ylabel(lbl, fontsize=9, rotation=90, labelpad=4)

fig.suptitle('Equivariance to rotation depends on filter symmetry', fontsize=12)
plt.tight_layout(); plt.show()"""))

G.append(md(r"""<a id="part3"></a>
# Part 3: Graphs and Permutation Symmetry

## 3.1 Why graphs?

Many real-world objects are naturally represented as graphs:
- **Molecules:** atoms = nodes (with element type), chemical bonds = edges.
- **Social networks:** users = nodes, friendships = edges.
- **Particle physics:** particles = nodes, interactions = edges.
- **Knowledge graphs:** entities = nodes, relations = edges.

A graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ has $n = |\mathcal{V}|$ nodes and $m = |\mathcal{E}|$ edges. Node features are stored as a matrix $\mathbf{H} \in \mathbb{R}^{n \times d}$, and the graph structure is captured by the **adjacency matrix** $A \in \{0,1\}^{n \times n}$.

**The key symmetry:** The labels we assign to nodes are arbitrary. Permuting the nodes — applying any $\sigma \in S_n$ — should not change the prediction (for a graph-level task) or should permute the predictions consistently (for a node-level task). The architecture must be **permutation equivariant** (for node-level tasks) or **permutation invariant** (for graph-level tasks)."""))

G.append(code(r"""# ── Draw some example graphs ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
graph_defs = [
    ('Benzene (C₆H₆)',
     list(range(6)),
     [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
     ['C']*6, '#3498db'),
    ('Karate club\n(social network)',
     list(range(8)),
     [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,7),(6,7),(1,2)],
     [str(i) for i in range(8)], '#e74c3c'),
    ('Knowledge graph\n(entity–relation)',
     ['Paris','France','EU','Macron','Person'],
     [('Paris','France'),('France','EU'),('Macron','France'),('Macron','Person'),('Paris','Person')],
     None, '#2ecc71'),
]

for ax, (title, nodes, edges, labels, color) in zip(axes, graph_defs):
    G_nx = nx.Graph()
    G_nx.add_nodes_from(nodes)
    G_nx.add_edges_from(edges)
    pos = nx.spring_layout(G_nx, seed=3)
    nx.draw_networkx_nodes(G_nx, pos, ax=ax, node_color=color,
                           node_size=400, alpha=0.85)
    nx.draw_networkx_edges(G_nx, pos, ax=ax, edge_color='#555555',
                           width=1.5, alpha=0.7)
    if labels:
        nx.draw_networkx_labels(G_nx, pos, ax=ax,
                                labels={n: l for n, l in zip(nodes, labels)},
                                font_size=9, font_color='white', font_weight='bold')
    else:
        nx.draw_networkx_labels(G_nx, pos, ax=ax, font_size=8)
    ax.set_title(title, fontsize=11); ax.axis('off')

fig.suptitle('Graphs arise naturally in chemistry, social networks, and knowledge representation',
             fontsize=12)
plt.tight_layout(); plt.show()"""))

G.append(md(r"""## 3.2 Message passing

The dominant framework for GNNs is **message passing** (Gilmer et al., 2017). Each node iteratively aggregates information from its neighbours:

$$\boxed{\mathbf{h}_v^{(k+1)} = \phi\!\left(\mathbf{h}_v^{(k)},\; \bigoplus_{u \in \mathcal{N}(v)} \psi\!\left(\mathbf{h}_v^{(k)},\, \mathbf{h}_u^{(k)}\right)\right),}$$

where:
- $\mathbf{h}_v^{(k)} \in \mathbb{R}^d$ is the feature vector of node $v$ at layer $k$,
- $\mathcal{N}(v)$ is the set of neighbours of $v$,
- $\psi$ is the **message function** (what each neighbour sends),
- $\bigoplus$ is a **permutation-invariant aggregation** (e.g., sum, mean, max),
- $\phi$ is the **update function** (combines own features with aggregated messages).

After $K$ layers, $\mathbf{h}_v^{(K)}$ captures information from the $K$-hop neighbourhood of $v$. A graph-level prediction uses a final **readout** $\mathbf{h}_{\mathcal{G}} = \mathrm{READOUT}(\{\mathbf{h}_v^{(K)}\}_{v \in \mathcal{V}})$, which must also be permutation-invariant (e.g., sum over all nodes)."""))

G.append(code(r"""# ── Animate message passing on a small molecule graph ───────────────────────
# Molecule: ethanol (CH3-CH2-OH), 3 heavy atoms simplified to a chain
# Nodes: 0=C, 1=C, 2=O   Features: [atomic_num, H_count]
node_feats_init = np.array([[6, 3],   # C, 3 H
                              [6, 2],   # C, 2 H
                              [8, 1]], dtype=float)  # O, 1 H
adj = np.array([[0,1,0],
                [1,0,1],
                [0,1,0]], dtype=float)   # chain C-C-O

def message_pass_step(H, A):
    # GCN-style: H_new = D^{-1/2} A~ D^{-1/2} H  (with self-loops)
    A_tilde = A + np.eye(len(A))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat @ H

node_names   = ['C (CH₃)', 'C (CH₂)', 'O (OH)']
feat_names   = ['atomic #', '# H']
colors_atoms = ['#3498db', '#3498db', '#e74c3c']

G_mol = nx.Graph()
G_mol.add_edges_from([(0,1),(1,2)])
pos_mol = {0: (0,0), 1: (1.5,0), 2: (3,0)}

H_layers = [node_feats_init.copy()]
H = node_feats_init.copy()
for _ in range(3):
    H = message_pass_step(H, adj)
    H_layers.append(H.copy())

def _plot_mp(layer=0):
    H_show = H_layers[layer]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0),
                              gridspec_kw={'width_ratios': [1, 1.5]})
    ax = axes[0]
    nx.draw_networkx_nodes(G_mol, pos_mol, ax=ax,
                           node_color=colors_atoms, node_size=900, alpha=0.9)
    nx.draw_networkx_edges(G_mol, pos_mol, ax=ax, edge_color='#555555', width=2.5)
    nx.draw_networkx_labels(G_mol, pos_mol, ax=ax,
                            labels={i: node_names[i] for i in range(3)},
                            font_size=8, font_color='white', font_weight='bold')
    # Show feature values above each node
    for i, (x, y) in pos_mol.items():
        ax.text(x, y + 0.35,
                f'[{H_show[i,0]:.2f}, {H_show[i,1]:.2f}]',
                ha='center', fontsize=9, color='#333333')
    ax.set_title(f'Layer {layer}: node features', fontsize=11)
    ax.set_xlim(-0.8, 3.8); ax.set_ylim(-0.8, 0.9); ax.axis('off')

    # Bar chart of features
    x_pos = np.arange(3)
    width = 0.35
    axes[1].bar(x_pos - width/2, H_show[:, 0], width, label='atomic #',
                color='#3498db', edgecolor='k', lw=0.5)
    axes[1].bar(x_pos + width/2, H_show[:, 1], width, label='# H',
                color='#e74c3c', edgecolor='k', lw=0.5)
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(node_names, fontsize=10)
    axes[1].set_title(f'Node features after {layer} message-passing steps', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].set_ylabel('Feature value')
    plt.tight_layout(); plt.show()

interact(_plot_mp,
         layer=IntSlider(min=0, max=3, step=1, value=0, description='Layer:'));"""))

G.append(md(r"""<a id="part4"></a>
# Part 4: Graph Convolutional Networks

## 4.1 GCN — spectral motivation

The **Graph Convolutional Network** (Kipf & Welling, 2017) derives its update rule from spectral graph theory. Define the normalised adjacency with self-loops:
$$\hat{A} = \tilde{D}^{-1/2}\tilde{A}\,\tilde{D}^{-1/2}, \qquad \tilde{A} = A + I, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}.$$

A GCN layer is:
$$\boxed{H^{(k+1)} = \sigma\!\left(\hat{A}\, H^{(k)}\, W^{(k)}\right),}$$
where $W^{(k)} \in \mathbb{R}^{d_k \times d_{k+1}}$ is a learnable weight matrix shared across all nodes.

**Permutation equivariance:** $\hat{A}$ and the multiplication $\hat{A} H W$ commute with any permutation $P$:
$$\hat{A}_{P} (P H) W = P (\hat{A} H W),$$
since permuting nodes permutes both $\hat{A}$ (via $P\hat{A}P^\top$) and $H$ (via $PH$) consistently. The layer is manifestly permutation-equivariant.

## 4.2 Spectral interpretation

The normalised Laplacian $\mathbf{L} = I - \hat{A}$ has eigenvectors $U$ (the graph Fourier basis). A spectral convolution with filter $\hat{g}$ is $U\,\mathrm{diag}(\hat{g})\,U^\top \mathbf{h}$. The GCN layer with $\hat{A} = I - \mathbf{L}$ corresponds to the first-order polynomial filter $\hat{g}(\lambda) = 1 - \lambda$ — a low-pass filter that smooths node features across the graph."""))

G.append(code(r"""# ── Node classification with a 2-layer GCN on a toy graph ───────────────────
rng3 = np.random.default_rng(42)

# Stochastic block model: 2 communities, 10 nodes each
n_per_class = 10
n_nodes = 2 * n_per_class
A = np.zeros((n_nodes, n_nodes))

for i in range(n_per_class):
    for j in range(i+1, n_per_class):
        if rng3.random() < 0.6:  # within-community edges
            A[i, j] = A[j, i] = 1.0
        if rng3.random() < 0.6:
            A[i+n_per_class, j+n_per_class] = A[j+n_per_class, i+n_per_class] = 1.0
for i in range(n_per_class):     # cross-community edges (sparse)
    for j in range(n_per_class):
        if rng3.random() < 0.05:
            A[i, j+n_per_class] = A[j+n_per_class, i] = 1.0

# Node features: noisy class indicator
H0 = np.zeros((n_nodes, 2))
H0[:n_per_class, 0]  = 1.0 + rng3.standard_normal(n_per_class) * 0.4
H0[n_per_class:, 1]  = 1.0 + rng3.standard_normal(n_per_class) * 0.4
labels = np.array([0]*n_per_class + [1]*n_per_class)

def gcn_layer(H, A, W, activation=True):
    A_tilde = A + np.eye(len(A))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    Z = A_hat @ H @ W
    return np.maximum(0, Z) if activation else Z  # ReLU

# Random weights (untrained — for illustration of message passing)
W1 = rng3.standard_normal((2, 8)) * 0.5
W2 = rng3.standard_normal((8, 2)) * 0.5

H1    = gcn_layer(H0, A, W1, activation=True)
H_out = gcn_layer(H1, A, W2, activation=False)
preds = np.argmax(H_out, axis=1)

G_sbm = nx.from_numpy_array(A)
pos_sbm = nx.spring_layout(G_sbm, seed=7)
node_colors = ['#3498db' if l == 0 else '#e74c3c' for l in labels]
pred_colors = ['#3498db' if p == 0 else '#e74c3c' for p in preds]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, cols, title in zip(axes,
        [node_colors, pred_colors],
        ['True community labels', 'GCN predictions (random weights)']):
    nx.draw_networkx_nodes(G_sbm, pos_sbm, ax=ax, node_color=cols,
                           node_size=200, alpha=0.9)
    nx.draw_networkx_edges(G_sbm, pos_sbm, ax=ax, edge_color='#aaaaaa',
                           width=0.8, alpha=0.5)
    ax.set_title(title, fontsize=11); ax.axis('off')

patches = [mpatches.Patch(color='#3498db', label='Community 0'),
           mpatches.Patch(color='#e74c3c', label='Community 1')]
axes[1].legend(handles=patches, fontsize=10)
fig.suptitle('Two-layer GCN on a stochastic block model graph', fontsize=12)
plt.tight_layout(); plt.show()
acc = (preds == labels).mean()
print(f"Accuracy with random weights: {acc:.0%}  (random chance: 50%)")
print("(Train the weights via gradient descent on cross-entropy to improve)")"""))

G.append(md(r"""<a id="part5"></a>
# Part 5: Expressivity — the Weisfeiler–Lehman Test

## 5.1 When can two graphs be distinguished?

A fundamental question: can a GNN distinguish non-isomorphic graphs? Two graphs are **isomorphic** if one can be obtained from the other by relabelling nodes. An ideal GNN would assign different representations to non-isomorphic graphs and the same representation to isomorphic ones.

**Key theorem** (Xu et al., 2019): *Any GNN using sum aggregation is at most as powerful as the Weisfeiler–Lehman (WL) graph isomorphism test.*

## 5.2 The WL test

The **1-WL test** assigns a colour (label) to each node and iteratively refines it:
1. **Initialise:** $c^{(0)}_v = h_v$ (initial node feature).
2. **Refine:** $c^{(k+1)}_v = \mathrm{HASH}\!\left(c^{(k)}_v,\; \{\!\{c^{(k)}_u : u \in \mathcal{N}(v)\}\!\}\right)$ (hash of own colour and *multiset* of neighbour colours).
3. **Compare:** at convergence, if the multisets of colours of two graphs differ, they are **not isomorphic**. If they agree, they *might* be isomorphic (WL cannot always decide).

The WL test fails on **regular graphs**: if all nodes have the same degree and the same multiset of neighbour degrees, WL assigns every node the same colour, regardless of global structure."""))

G.append(code(r"""def wl_step(colors, adj_list):
    new_colors = {}
    for v, c in colors.items():
        nbr_colors = tuple(sorted(colors[u] for u in adj_list[v]))
        new_colors[v] = hash((c, nbr_colors)) % (10**6)
    return new_colors

def run_wl(G_nx, init_color=None, n_steps=4):
    adj_list = {v: list(G_nx.neighbors(v)) for v in G_nx.nodes()}
    if init_color is None:
        colors = {v: 0 for v in G_nx.nodes()}
    else:
        colors = dict(init_color)
    history = [dict(colors)]
    for _ in range(n_steps):
        colors = wl_step(colors, adj_list)
        history.append(dict(colors))
    return history

# Two non-isomorphic graphs that WL cannot distinguish (both 3-regular, 6 nodes)
G1 = nx.Graph([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4),(2,5)])  # K_{3,3}
G2 = nx.Graph([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)])  # prism graph

wl1 = run_wl(G1)
wl2 = run_wl(G2)

def _plot_wl(step=0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.Set1
    pos1 = nx.circular_layout(G1)
    pos2 = nx.spring_layout(G2, seed=2)

    for ax, Gx, wl, pos, title in [
            (axes[0], G1, wl1, pos1, 'Graph 1  (K₃,₃)'),
            (axes[1], G2, wl2, pos2, 'Graph 2  (Prism)'),
    ]:
        cols = wl[min(step, len(wl)-1)]
        unique = sorted(set(cols.values()))
        color_map = {c: cmap(i / max(1, len(unique)-1)) for i, c in enumerate(unique)}
        node_cols  = [color_map[cols[v]] for v in Gx.nodes()]

        nx.draw_networkx_nodes(Gx, pos, ax=ax, node_color=node_cols,
                               node_size=500, alpha=0.9)
        nx.draw_networkx_edges(Gx, pos, ax=ax, edge_color='#555555', width=1.5)
        nx.draw_networkx_labels(Gx, pos, ax=ax,
                                labels={v: str(cols[v])[-3:] for v in Gx.nodes()},
                                font_size=7, font_color='white')
        multiset = sorted(cols.values())
        ax.set_title(f'{title}\nColour multiset: {multiset}', fontsize=10)
        ax.axis('off')

    fig.suptitle(f'Weisfeiler–Lehman test — step {step}  '
                 '(same colour multisets → WL cannot distinguish)', fontsize=11)
    plt.tight_layout(); plt.show()

interact(_plot_wl,
         step=IntSlider(min=0, max=4, step=1, value=0, description='Step:'));"""))

G.append(md(r"""> **Exercise 5.1.** *(WL and GNN expressivity)*
>
> **(a)** Verify by running `run_wl` that the colour multisets of $G_1$ and $G_2$ remain identical at every WL step. Confirm that the graphs are *not* isomorphic by checking their degree sequences and cycle structure.
>
> **(b)** What aggregation function $\bigoplus$ gives GNNs strictly greater expressivity than WL? (Hint: see Xu et al. 2019, Theorem 3.) Why does *mean* aggregation fail where *sum* succeeds?
>
> **(c)** Higher-order WL tests ($k$-WL for $k \ge 2$) are strictly more powerful. Describe informally what $2$-WL considers that $1$-WL does not. What is the computational cost?"""))

G.append(md(r"""<a id="part6"></a>
# Part 6: Beyond 2D — Equivariance in 3D Science

## 6.1 Molecules and the Euclidean group

A molecule is a set of atoms, each with a position $\mathbf{r}_i \in \mathbb{R}^3$ and an atom type $z_i$. A prediction of a physical property (energy, reactivity, binding affinity) must be:
- **Invariant to rotation and reflection:** $E(R\{\mathbf{r}_i\}) = E(\{\mathbf{r}_i\})$ for $R \in O(3)$.
- **Invariant to translation:** $E(\{\mathbf{r}_i + \mathbf{t}\}) = E(\{\mathbf{r}_i\})$.
- **Invariant to permutation of atoms** with the same type.

The relevant symmetry group is $E(3) = \mathbb{R}^3 \rtimes O(3)$ (rotations, reflections, translations). Architectures that respect this are called **E(3)-equivariant GNNs**.

**Examples:**
- **SchNet** (Schütt et al., 2017): uses pairwise distances $\|\mathbf{r}_i - \mathbf{r}_j\|$ as edge features — these are $E(3)$-invariant.
- **DimeNet** (Gasteiger et al., 2020): additionally uses bond angles $\theta_{ijk}$ — also $E(3)$-invariant.
- **SE(3)-Transformers** / **EGNN** (Satorras et al., 2021): pass *equivariant* vector features alongside invariant scalar features.
- **AlphaFold 2** (Jumper et al., 2021): uses frames (local coordinate systems per residue) that transform equivariantly — a key ingredient in predicting protein structure to near-experimental accuracy.

## 6.2 The blueprint of geometric deep learning

Bronstein et al. (2021) show that the following five architecture families are all instances of the same symmetry-based blueprint:

| Domain | Symmetry group $G$ | Architecture |
|--------|--------------------|-----------|
| Grids / images | $(\mathbb{Z}^2, +)$ translation | **CNN** |
| Sets | $S_n$ permutation | **Deep Sets** |
| Graphs | $S_n$ permutation | **GNN** |
| Groups | $G$ (any) | **Group CNN** |
| Manifolds / meshes | Diffeomorphism group | **Intrinsic mesh CNNs** |
| 3-D point clouds | $E(3)$ / $SE(3)$ | **E(n)-equivariant GNNs** |

The same mathematical question — *"what linear maps commute with the group action?"* — determines the layer structure in every case."""))

G.append(code(r"""# ── Visualise: distance-based features are E(3)-invariant ───────────────────
rng4 = np.random.default_rng(5)

# Small toy molecule: 4 atoms in 3D
positions = np.array([[0,0,0],[1.5,0,0],[0.75,1.3,0],[0.75,0.43,1.22]])
atom_types = ['C','C','C','H']

# Apply a random rotation R ∈ SO(3)
from scipy.spatial.transform import Rotation
R_rand = Rotation.random(random_state=3).as_matrix()
pos_rot = positions @ R_rand.T

# Pairwise distances before and after rotation
def pairwise_dists(pos):
    n = len(pos)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(pos[i] - pos[j])
    return D

D_orig = pairwise_dists(positions)
D_rot  = pairwise_dists(pos_rot)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, D, title in [(axes[0], D_orig, 'Distance matrix\n(original positions)'),
                      (axes[1], D_rot,  'Distance matrix\n(after random rotation)'),
                      (axes[2], np.abs(D_orig - D_rot), r'$|D_\mathrm{orig} - D_\mathrm{rot}|$'+'\n(should be ≈ 0)')]:
    im = ax.imshow(D, cmap='Blues', vmin=0)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(atom_types); ax.set_yticklabels(atom_types)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{D[i,j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle(r'Pairwise distances are $E(3)$-invariant: identical before and after rotation',
             fontsize=12)
plt.tight_layout(); plt.show()
print(f"Max change in distance matrix after rotation: {np.abs(D_orig - D_rot).max():.2e}")"""))

G.append(md(r"""---
# Summary

| Concept | Key formula / takeaway |
|---------|------------------------|
| **Equivariance** | $f(g\cdot\mathbf{x}) = g\cdot f(\mathbf{x})$ for all $g\in G$ |
| **CNN** | Equivariant to $(\mathbb{Z}^2,+)$ by construction (via convolution) |
| **Message passing** | $\mathbf{h}_v^{(k+1)} = \phi(\mathbf{h}_v^{(k)}, \bigoplus_{u\in\mathcal{N}(v)}\psi(\mathbf{h}_v^{(k)},\mathbf{h}_u^{(k)}))$ |
| **GCN layer** | $H^{(k+1)} = \sigma(\hat{A}H^{(k)}W^{(k)})$,  $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ |
| **Permutation equivariance** | Sum/mean aggregation is permutation-invariant; node updates are equivariant |
| **WL expressivity limit** | Any sum-GNN $\le$ WL test; WL fails on regular graphs |
| **3-D equivariance** | $E(3)$-equivariant GNNs use distances/angles as invariant features |"""))

# ════════════════════════════════════════════════════════════════════════════
# SAVE ALL THREE NOTEBOOKS
# ════════════════════════════════════════════════════════════════════════════

base = pathlib.Path('/Users/eduenez/repos/MathAIspring2026UTSA/code')
files = [
    (base / 'diffusion_models.ipynb',        D),
    (base / 'reinforcement_learning.ipynb',  R),
    (base / 'geometric_deep_learning.ipynb', G),
]
for path, cells in files:
    path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False))
    ct = sum(1 for c in cells if c['cell_type'] == 'code')
    mt = sum(1 for c in cells if c['cell_type'] == 'markdown')
    print(f"Written: {path.name}  ({len(cells)} cells: {mt} md, {ct} code)")
