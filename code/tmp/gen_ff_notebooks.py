#!/usr/bin/env python3
"""Generate fwdpass_units.ipynb and backpropagation.ipynb."""
import json, itertools, pathlib

_ctr = itertools.count(1)

def mk_id():
    return f"c{next(_ctr):06d}"

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
# NOTEBOOK 1 — fwdpass_units.ipynb
# ════════════════════════════════════════════════════════════════════════════

C1 = []

C1.append(md(r"""# Feedforward Networks I: The Forward Pass and Activation Functions
## Units, nonlinearities, and how information flows through a network

**MAT 4953/6973 — Mathematical Foundations of AI** (Spring 2026, UTSA)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/fwdpass_units.ipynb)

---"""))

C1.append(md(r"""> **Prerequisites:** Vectors and matrices (matrix–vector products, norms), basic calculus (derivatives, chain rule), Python/NumPy. Familiarity with [`intro_feedforward_mnist.ipynb`](intro_feedforward_mnist.ipynb) is assumed.

In [`intro_feedforward_mnist.ipynb`](intro_feedforward_mnist.ipynb) we built a feedforward network and trained it on MNIST using Keras.
We wrote `layers.Dense(64, activation="relu")` without stopping to ask:

- What exactly *is* a unit (neuron)?
- Why ReLU and not something else — sigmoid, tanh, GeLU?
- How do the weight matrices $W^{(l)}$ and biases $\mathbf{b}^{(l)}$ combine to turn input $\mathbf{x}$ into output $\hat{\mathbf{y}}$?

This notebook answers those questions. The **forward pass** — computing $\hat{\mathbf{y}}$ from $\mathbf{x}$ — is the foundation for everything else: loss evaluation, gradient computation, and training.

**Outline**
1. [Feedforward architecture — a quick recap](#part1)
2. [The forward pass, layer by layer](#part2)
3. [Activation functions — a tour of units](#part3)
4. [Tracing a complete forward pass](#part4)"""))

C1.append(code(r"""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ipywidgets import interact, FloatSlider, Dropdown
from scipy.special import ndtr          # standard normal CDF, used for GeLU
from scipy.stats import norm as _norm   # for GeLU derivative

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'figure.dpi': 100})
rng = np.random.default_rng(42)
print("Libraries loaded.")"""))

C1.append(md(r"""<a id="part1"></a>
# Part 1: Feedforward Architecture — A Quick Recap

## 1.1 Layers, weights, and biases

A **feedforward network** (or **multi-layer perceptron**, MLP) is an ordered sequence of layers:

$$\underbrace{\mathbf{x}}_{\text{input}} \;\longrightarrow\; \underbrace{\text{layer 1}}_{\text{hidden}} \;\longrightarrow\; \cdots \;\longrightarrow\; \underbrace{\text{layer } (L-1)}_{\text{hidden}} \;\longrightarrow\; \underbrace{\hat{\mathbf{y}}}_{\text{output}}$$

Each layer $l \in \{1, \ldots, L\}$ has **width** $n_l$ (number of units). Layer $l-1$ connects to layer $l$ through:
- a **weight matrix** $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$, and
- a **bias vector** $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$.

In the MNIST model we used the architecture $(n_0, n_1, n_2, n_3) = (784, 64, 64, 10)$.
The figure below shows a scaled-down version with the same structure."""))

C1.append(code(r"""def draw_network(layer_sizes, ax, xs=2.4, ys=1.05, r=0.27, max_n=8):
    # Draw a feedforward network diagram on ax.
    L = len(layer_sizes)
    node_pos = {}
    clr  = ['#d5e8d4'] + ['#dae8fc'] * (L - 2) + ['#fff2cc']
    eclr = ['#82b366'] + ['#6c8ebf'] * (L - 2) + ['#d6b656']
    for l, n in enumerate(layer_sizes):
        shown = min(n, max_n)
        ys_arr = np.linspace(-(shown - 1) / 2 * ys, (shown - 1) / 2 * ys, shown)
        x = l * xs
        for j, y in enumerate(ys_arr):
            node_pos[(l, j)] = (x, y)
            ax.add_patch(plt.Circle((x, y), r, color=clr[l], ec=eclr[l], lw=1.5, zorder=3))
        if n > max_n:
            ax.text(x, -(shown) / 2 * ys - 0.15, r'$\vdots$',
                    ha='center', va='top', fontsize=13)
    for l in range(L - 1):
        for j in range(min(layer_sizes[l], max_n)):
            for k in range(min(layer_sizes[l + 1], max_n)):
                x0, y0 = node_pos[(l, j)]
                x1, y1 = node_pos[(l + 1, k)]
                ax.plot([x0 + r, x1 - r], [y0, y1], 'k-', alpha=0.10, lw=0.7, zorder=1)
    return node_pos

fig, ax = plt.subplots(figsize=(10, 5))
pos = draw_network([4, 5, 5, 3], ax)
ax.set_xlim(-0.7, 3 * 2.4 + 0.7)
ax.set_ylim(-3.5, 4.2)
ax.set_aspect('equal')
ax.axis('off')

layer_labels = [r'Input$\;\mathbf{x}$'+'\n'+r'$n_0 = 4$',
                r'Hidden 1'+'\n'+r'$n_1 = 5$',
                r'Hidden 2'+'\n'+r'$n_2 = 5$',
                r'Output$\;\hat{\mathbf{y}}$'+'\n'+r'$n_3 = 3$']
for l, lbl in enumerate(layer_labels):
    ax.text(l * 2.4, 3.6, lbl, ha='center', fontsize=11)

patches = [mpatches.Patch(fc='#d5e8d4', ec='#82b366', label='Input layer'),
           mpatches.Patch(fc='#dae8fc', ec='#6c8ebf', label='Hidden layers'),
           mpatches.Patch(fc='#fff2cc', ec='#d6b656', label='Output layer')]
ax.legend(handles=patches, loc='lower right', fontsize=10)
ax.set_title('Feedforward network with 2 hidden layers  (scaled-down MNIST architecture)',
             fontsize=12)
plt.tight_layout()
plt.show()"""))

C1.append(md(r"""## 1.2 What is a single unit?

Each **unit** $k$ in layer $l$ computes its output in two steps:

**Step 1 — Pre-activation** (linear):
$$z_k^{(l)} = \sum_{j=1}^{n_{l-1}} W^{(l)}_{kj}\, a_j^{(l-1)} + b_k^{(l)}.$$

**Step 2 — Activation** (nonlinear):
$$a_k^{(l)} = \sigma\!\left(z_k^{(l)}\right).$$

Written for all $n_l$ units in layer $l$ simultaneously (using matrix–vector notation):

$$\boxed{\mathbf{z}^{(l)} = W^{(l)}\,\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \sigma\!\left(\mathbf{z}^{(l)}\right).}$$

Here $\sigma$ is applied **element-wise**: $\sigma(\mathbf{z})_k = \sigma(z_k)$.
The choice of $\sigma$ defines the *type* of unit; [Part 3](#part3) surveys the most important choices."""))

C1.append(md(r"""<a id="part2"></a>
# Part 2: The Forward Pass

## 2.1 Layer by layer

Setting $\mathbf{a}^{(0)} = \mathbf{x}$, the network applies:
$$\mathbf{z}^{(l)} = W^{(l)}\,\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \sigma^{(l)}\!\left(\mathbf{z}^{(l)}\right), \qquad l = 1, \ldots, L.$$

The output is $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$.
The full computation is the composition $F^{(L)} \circ \cdots \circ F^{(1)}$ where each *layer map* is:
$$F^{(l)}(\mathbf{a}) = \sigma^{(l)}\!\left(W^{(l)}\mathbf{a} + \mathbf{b}^{(l)}\right).$$
This is exactly what `model.predict(x)` does in `intro_feedforward_mnist.ipynb`.

## 2.2 Why nonlinearity is indispensable

**Claim:** if every $\sigma^{(l)}$ is the identity ($\mathbf{a}^{(l)} = \mathbf{z}^{(l)}$), the entire $L$-layer network collapses to a *single* affine map.

**Proof:** with $\sigma = \mathrm{Id}$ we can substitute layer by layer:
$$\hat{\mathbf{y}} = W^{(L)}\!\left(\cdots\!\left(W^{(2)}\!\left(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}\right) + \mathbf{b}^{(2)}\right)\!\cdots\right) + \mathbf{b}^{(L)} = \underbrace{W^{(L)}\cdots W^{(1)}}_{\widetilde{W}}\,\mathbf{x} + \widetilde{\mathbf{b}}.$$

Depth without nonlinearity adds no representational power — the network can only learn linear functions.
The code below lets you verify this empirically."""))

C1.append(code(r"""# Verify: stacking linear layers = one linear layer
rng2 = np.random.default_rng(7)
n0, n1, n2 = 3, 5, 2

W1 = rng2.standard_normal((n1, n0))
b1 = rng2.standard_normal(n1)
W2 = rng2.standard_normal((n2, n1))
b2 = rng2.standard_normal(n2)
x  = rng2.standard_normal(n0)

# Two-layer forward pass, identity activation
a1 = W1 @ x + b1        # z1 = a1 (no nonlinearity)
y_deep = W2 @ a1 + b2   # z2 = y_deep

# Equivalent single layer
W_tilde = W2 @ W1
b_tilde = W2 @ b1 + b2
y_flat  = W_tilde @ x + b_tilde

print(f"Two-layer output:  {y_deep}")
print(f"Single-layer output: {y_flat}")
print(f"Are they equal?    {np.allclose(y_deep, y_flat)}")"""))

C1.append(md(r"""> **Exercise 2.1.** *(Linear networks collapse)*
>
> **(a)** The code above confirms that two linear layers equal one linear layer. Extend the scaffold to $L = 5$ layers and verify the same property holds.
>
> **(b)** Now suppose $\sigma(z) = 2z + 1$ (a linear activation, not the identity). Show algebraically — and verify in code — that the network is still equivalent to a single affine map. What are $\widetilde{W}$ and $\widetilde{\mathbf{b}}$ in this case?
>
> **(c)** *(Challenge)* Is the class of functions representable by a network with $L$ *nonlinear* layers strictly larger than the class representable by $L-1$ nonlinear layers? What does the **universal approximation theorem** say?"""))

C1.append(md(r"""<a id="part3"></a>
# Part 3: Activation Functions — A Tour of Units

## 3.1 Hidden-layer activations

The activation function determines what each hidden unit "detects" and, crucially, how gradients flow backward during training (more on that in [`backpropagation.ipynb`](backpropagation.ipynb)).

| Name | Formula | Range | Key virtue | Caution |
|------|---------|-------|-----------|---------|
| **Sigmoid** | $\dfrac{1}{1+e^{-z}}$ | $(0,1)$ | Smooth; natural probability interpretation | Saturates for large $|z|$; vanishing gradients |
| **Tanh** | $\tanh(z)$ | $(-1,1)$ | Zero-centred; stronger gradients near 0 | Still saturates at extremes |
| **ReLU** | $\max(0,z)$ | $[0,\infty)$ | Fast; sparse activations; no saturation for $z>0$ | "Dying ReLU": unit stuck at 0 if always $z\le 0$ |
| **Leaky ReLU** | $\max(\alpha z,\,z),\;\alpha\ll 1$ | $\mathbb{R}$ | Fixes dying ReLU with small gradient for $z<0$ | Extra hyperparameter $\alpha$ |
| **GeLU** | $z\,\Phi(z)$ | $\mathbb{R}$ | Smooth; state-of-the-art in transformers (BERT, GPT) | Slightly costlier to evaluate |

Here $\Phi(z) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z} e^{-t^2/2}\,dt$ is the standard normal CDF.

**A note on GeLU.** $\text{GeLU}(z) = z \cdot \Phi(z)$ can be read as *soft stochastic gating*: the input $z$ is multiplied by the probability that a standard normal $N \le z$. For $z \gg 0$ this is $\approx z$ (acts like the identity, same as ReLU); for $z \ll 0$ it is $\approx 0$ (same as ReLU). Unlike ReLU, GeLU is smooth everywhere — its derivative exists at $z=0$ — which is believed to help optimisation.

## 3.2 Output-layer units: Softmax

For multi-class classification the output layer uses **softmax**, which turns a vector of *logits* $\mathbf{z}^{(L)} \in \mathbb{R}^K$ into a probability distribution over $K$ classes:

$$\boxed{\hat{y}_k = \mathrm{softmax}(\mathbf{z}^{(L)})_k = \frac{e^{z_k^{(L)}}}{\displaystyle\sum_{j=1}^{K} e^{z_j^{(L)}}}, \qquad k = 1, \ldots, K.}$$

The outputs satisfy $\hat{y}_k \ge 0$ and $\sum_k \hat{y}_k = 1$.
In `intro_feedforward_mnist.ipynb` the output layer had $K = 10$ (one per digit class); Keras's `SparseCategoricalCrossentropy(from_logits=True)` applies softmax internally, which is why the final `Dense(10)` layer had no explicit activation."""))

C1.append(code(r"""# Activation function definitions
z = np.linspace(-4.5, 4.5, 500)

def sigmoid(z):    return 1 / (1 + np.exp(-z))
def d_sigmoid(z):  s = sigmoid(z); return s * (1 - s)
def relu(z):       return np.maximum(0, z)
def d_relu(z):     return (z > 0).astype(float)
def lrelu(z, a=0.1): return np.where(z > 0, z, a * z)
def d_lrelu(z, a=0.1): return np.where(z > 0, 1.0, a)
def gelu(z):       return z * ndtr(z)
def d_gelu(z):     return ndtr(z) + z * _norm.pdf(z)

acts = [
    ('Sigmoid',     sigmoid, d_sigmoid, '#3498db'),
    ('Tanh',        np.tanh, lambda z: 1 - np.tanh(z)**2, '#e74c3c'),
    ('ReLU',        relu,    d_relu,    '#2ecc71'),
    ('Leaky ReLU',  lrelu,   d_lrelu,   '#f39c12'),
    ('GeLU',        gelu,    d_gelu,    '#9b59b6'),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for name, fn, dfn, c in acts:
    axes[0].plot(z, fn(z),  color=c, lw=2.2, label=name)
    axes[1].plot(z, dfn(z), color=c, lw=2.2, label=name)

for ax, title in zip(axes, [r'Activation functions $\sigma(z)$',
                              r"Derivatives $\sigma'(z)$"]):
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.axvline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel(r'$z$', fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-4.5, 4.5)

axes[0].set_ylim(-1.3, 2.0)
axes[1].set_ylim(-0.15, 1.2)
plt.tight_layout()
plt.show()"""))

C1.append(md(r"""**Observations to make:**
- **Sigmoid and Tanh** both saturate: their derivatives $\to 0$ as $|z| \to \infty$. A unit in this regime receives almost no gradient signal — the **vanishing gradient** problem.
- **ReLU** has derivative exactly 1 for $z > 0$ (no saturation), but exactly 0 for $z < 0$ (potential dead units).
- **Leaky ReLU** keeps a small slope $\alpha$ for $z < 0$, preventing dead units.
- **GeLU** is smooth and interpolates between 0 and $z$; its derivative is positive everywhere (no dead units) but less than 1 near $z = 0$ (some saturation-like behaviour).

**Experiment:** The interactive below highlights one activation against the ReLU baseline. Pay attention to:
- Where does the derivative approach zero (vanishing gradient region)?
- Is the function differentiable at $z = 0$?"""))

C1.append(code(r"""def _plot_one_act(activation='GeLU', show_derivative=True):
    fn_map = {'Sigmoid': (sigmoid, d_sigmoid, '#3498db'),
              'Tanh':    (np.tanh, lambda z: 1 - np.tanh(z)**2, '#e74c3c'),
              'ReLU':    (relu,    d_relu,    '#2ecc71'),
              'Leaky ReLU': (lrelu, d_lrelu, '#f39c12'),
              'GeLU':    (gelu,    d_gelu,    '#9b59b6')}
    fn, dfn, color = fn_map[activation]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(z, relu(z), color='#cccccc', lw=1.5, ls='--', label='ReLU (reference)')
    ax.plot(z, fn(z), color=color, lw=2.5, label=activation)
    if show_derivative:
        ax.plot(z, dfn(z), color=color, lw=2, ls=':', alpha=0.8,
                label=f"{activation}' (derivative)")
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$z$', fontsize=13)
    ax.set_ylim(-1.4, 2.3)
    ax.set_xlim(-4.5, 4.5)
    ax.legend(fontsize=11)
    ax.set_title(f'{activation} vs ReLU (dashed = derivative)', fontsize=13)
    plt.tight_layout(); plt.show()

interact(_plot_one_act,
         activation=Dropdown(options=['Sigmoid','Tanh','ReLU','Leaky ReLU','GeLU'],
                              value='GeLU', description='Activation:'),
         show_derivative=True);"""))

C1.append(md(r"""## 3.3 Softmax in action

Softmax converts logits into probabilities. Drag the logit sliders to see how the probability mass shifts.
Notice the **sharpening effect**: increasing one logit relative to the others concentrates nearly all probability mass on that class."""))

C1.append(code(r"""def _softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

def _plot_softmax(z1=2.0, z2=0.5, z3=-0.5, z4=-1.5, z5=-2.0):
    logits = np.array([z1, z2, z3, z4, z5])
    probs  = _softmax(logits)
    labels = [f'C{k+1}\n$z={v:.1f}$' for k, v in enumerate(logits)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(labels, logits, color=colors, edgecolor='k', lw=0.5)
    axes[0].axhline(0, color='k', lw=0.6)
    axes[0].set_ylabel(r'Logit $z_k$')
    axes[0].set_title(r'Input logits $\mathbf{z}$')

    bars = axes[1].bar(labels, probs, color=colors, edgecolor='k', lw=0.5)
    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, p + 0.01,
                     f'{p:.3f}', ha='center', fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel(r'Probability $\hat{y}_k$')
    axes[1].set_title(r'Softmax output $\hat{\mathbf{y}}$')
    plt.tight_layout(); plt.show()

interact(_plot_softmax,
         z1=FloatSlider(value=2.0,  min=-4, max=4, step=0.1, description=r'$z_1$:'),
         z2=FloatSlider(value=0.5,  min=-4, max=4, step=0.1, description=r'$z_2$:'),
         z3=FloatSlider(value=-0.5, min=-4, max=4, step=0.1, description=r'$z_3$:'),
         z4=FloatSlider(value=-1.5, min=-4, max=4, step=0.1, description=r'$z_4$:'),
         z5=FloatSlider(value=-2.0, min=-4, max=4, step=0.1, description=r'$z_5$:'));"""))

C1.append(md(r"""<a id="part4"></a>
# Part 4: Tracing a Forward Pass End-to-End

Let us trace a complete forward pass through a concrete small network: architecture $(3 \to 4 \to 2)$, ReLU hidden units, softmax output.

$$\mathbf{x} \in \mathbb{R}^3
  \;\xrightarrow{W^{(1)},\,\mathbf{b}^{(1)}}\;
  \mathbf{z}^{(1)} \in \mathbb{R}^4
  \;\xrightarrow{\mathrm{ReLU}}\;
  \mathbf{a}^{(1)} \in \mathbb{R}^4
  \;\xrightarrow{W^{(2)},\,\mathbf{b}^{(2)}}\;
  \mathbf{z}^{(2)} \in \mathbb{R}^2
  \;\xrightarrow{\mathrm{softmax}}\;
  \hat{\mathbf{y}} \in \mathbb{R}^2$$

This is the same structure as the MNIST model, just with $n_0 = 3$ (instead of 784) and $n_2 = 2$ (instead of 10)."""))

C1.append(code(r"""np.set_printoptions(precision=4, suppress=True)
rng3 = np.random.default_rng(0)

# Random network parameters (illustrative)
W1 = rng3.standard_normal((4, 3)) * 0.5
b1 = np.zeros(4)
W2 = rng3.standard_normal((2, 4)) * 0.5
b2 = np.zeros(2)

x = np.array([0.5, -1.2, 0.8])   # input vector

sep = '=' * 52
print(sep)
print('FORWARD PASS   network shape  (3 → 4 → 2)')
print(sep)

# --- Layer 1 ---
a0 = x
z1 = W1 @ a0 + b1
a1 = relu(z1)
print(f'\nLayer 1  (ReLU hidden,  n₁ = 4)')
print(f'  z⁽¹⁾ = W⁽¹⁾ x + b⁽¹⁾  =  {z1}')
print(f'  a⁽¹⁾ = ReLU(z⁽¹⁾)    =  {a1}')
print(f'  units zeroed by ReLU:  {(z1 < 0).sum()} of {len(z1)}')

# --- Layer 2 ---
z2 = W2 @ a1 + b2
y_hat = _softmax(z2)
print(f'\nLayer 2  (softmax output,  n₂ = 2)')
print(f'  z⁽²⁾ = W⁽²⁾ a⁽¹⁾ + b⁽²⁾  =  {z2}')
print(f'  ŷ = softmax(z⁽²⁾)      =  {y_hat}')
print(f'  Σ ŷₖ = {y_hat.sum():.6f}   (must equal 1)')"""))

C1.append(md(r"""# Exercises

> **Exercise 4.1.** *(Counting parameters)*
>
> **(a)** Derive a formula for the total number of trainable parameters (weights + biases) in a network with architecture $(n_0, n_1, \ldots, n_L)$.
>
> **(b)** Apply your formula to $(784, 64, 64, 10)$ and verify it matches the $55{,}050$ reported in `intro_feedforward_mnist.ipynb`.
>
> **(c)** If both hidden layers are widened to 128 units — architecture $(784, 128, 128, 10)$ — by what factor does the total parameter count increase?

> **Exercise 4.2.** *(Choosing activations)*
>
> For each task below, state the appropriate output activation and explain why.
>
> **(a)** Classify an image as exactly one of $K = 1{,}000$ categories.
>
> **(b)** Detect which of $K = 10$ attributes are present in an image (multiple attributes can be present simultaneously).
>
> **(c)** Predict a continuous quantity such as house price or temperature.

> **Exercise 4.3.** *(Vanishing gradients — experiment)*
>
> The scaffold below propagates a gradient signal backward through a deep network and records its norm at each layer. Run it with `act = 'sigmoid'` and then with `act = 'relu'`.
>
> **(a)** Describe what happens to the gradient norm as you go deeper with sigmoid. Why?
>
> **(b)** Does the same phenomenon occur with ReLU? Explain the difference using the derivative plots from Part 3.
>
> **(c)** *(Challenge)* Modify the weight initialisation (currently `scale = 0.5`) to mitigate the vanishing gradient for sigmoid. What scale makes the gradient norms roughly constant across layers?"""))

C1.append(code(r"""depth  = 10    # @param {type:"integer"}
width  = 40    # @param {type:"integer"}
act    = 'sigmoid'   # @param ["sigmoid", "relu", "tanh", "gelu"]
scale  = 0.5   # @param {type:"number"}

_act  = {'sigmoid': sigmoid, 'relu': relu,
         'tanh': np.tanh,   'gelu': gelu}[act]
_dact = {'sigmoid': d_sigmoid, 'relu': d_relu,
         'tanh': lambda z: 1 - np.tanh(z)**2,
         'gelu': d_gelu}[act]

rng4 = np.random.default_rng(1)
Ws = [rng4.standard_normal((width, width)) * scale for _ in range(depth)]
x0 = rng4.standard_normal(width)

# Forward pass
zs, as_ = [], [x0]
for W in Ws:
    zs.append(W @ as_[-1])
    as_.append(_act(zs[-1]))

# Backward pass  (loss = sum of outputs; upstream seed = ones)
grad = np.ones(width)
norms = []
for l in reversed(range(depth)):
    grad = grad * _dact(zs[l])   # local gradient through activation
    grad = Ws[l].T @ grad        # local gradient through linear map
    norms.insert(0, np.linalg.norm(grad))

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(range(1, depth + 1), norms, 'o-', color='#3498db', lw=2, ms=5)
ax.set_xlabel('Layer depth (1 = closest to output)')
ax.set_ylabel('Gradient norm  (log scale)')
ax.set_title(f'Gradient signal through depth — activation: {act},  init scale: {scale}')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()"""))

C1.append(md(r"""---
# Summary

| Concept | Key formula / takeaway |
|---------|------------------------|
| **Unit (pre-activation)** | $\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ |
| **Unit (activation)** | $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$, element-wise |
| **No nonlinearity** | Stacking linear layers $=$ one linear layer |
| **Sigmoid / Tanh** | Smooth, bounded; saturate for large $|z|$ $\Rightarrow$ vanishing gradients |
| **ReLU** | $\max(0,z)$; fast, sparse; risk of dying units |
| **GeLU** | $z\,\Phi(z)$; smooth; standard in transformer architectures |
| **Softmax** | Vector output: $\hat{y}_k = e^{z_k}/\sum_j e^{z_j}$, sums to 1 |

**Next:** [`backpropagation.ipynb`](backpropagation.ipynb) shows how to compute $\partial\mathcal{L}/\partial W^{(l)}$ for every weight in the network — efficiently, in a single backward pass."""))

# ════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — backpropagation.ipynb
# ════════════════════════════════════════════════════════════════════════════

C2 = []

C2.append(md(r"""# Feedforward Networks II: The Backpropagation Algorithm
## Chain rule, computational graphs, and gradient flow

**MAT 4953/6973 — Mathematical Foundations of AI** (Spring 2026, UTSA)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/backpropagation.ipynb)

---"""))

C2.append(md(r"""> **Prerequisites:** [`fwdpass_units.ipynb`](fwdpass_units.ipynb) (forward pass, activation functions), multivariate derivatives (partial derivatives, chain rule), Python/NumPy.

Training a network requires computing $\partial\mathcal{L}/\partial W^{(l)}$ and $\partial\mathcal{L}/\partial\mathbf{b}^{(l)}$ for **every** weight and bias across all layers, at every training step.
For the MNIST model there are 55,050 parameters — so we need 55,050 partial derivatives, repeated for every mini-batch.

**The naive approach (finite differences)** estimates each derivative as:
$$\frac{\partial\mathcal{L}}{\partial w_i} \approx \frac{\mathcal{L}(\mathbf{w} + \varepsilon\mathbf{e}_i) - \mathcal{L}(\mathbf{w})}{\varepsilon},$$
but requires one full forward pass per parameter. For $P = 55{,}050$ parameters that is $55{,}050$ forward passes per gradient step — completely impractical.

**Backpropagation** exploits the chain rule to compute *all* $P$ partial derivatives in a *single* backward pass — the same cost as one forward pass. This is its fundamental miracle.

**Outline**
1. [The chain rule of calculus](#part1)
2. [Computational graphs](#part2)
3. [Backward pass: upstream × local](#part3)
4. [Backpropagation in a feedforward network](#part4)
5. [NumPy implementation and gradient check](#part5)"""))

C2.append(code(r"""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from ipywidgets import interact, IntSlider
from scipy.special import ndtr
from scipy.stats import norm as _norm

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'figure.dpi': 100})

# Activation functions (same as fwdpass_units.ipynb)
def sigmoid(z):    return 1 / (1 + np.exp(-z))
def d_sigmoid(z):  s = sigmoid(z); return s * (1 - s)
def relu(z):       return np.maximum(0, z)
def d_relu(z):     return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - np.max(z)); return e / e.sum()
def cross_entropy(y_hat, y_true):
    return -np.sum(y_true * np.log(y_hat + 1e-12))

print("Libraries loaded.")"""))

C2.append(md(r"""<a id="part1"></a>
# Part 1: The Chain Rule of Calculus

## 1.1 Single-variable chain rule

If $a = f(z)$ and $z = g(x)$, then $a = f(g(x))$ and:

$$\boxed{\frac{da}{dx} = \frac{da}{dz}\cdot\frac{dz}{dx}.}$$

**Example:** $a = \sigma(z) = \frac{1}{1+e^{-z}}$ and $z = wx + b$:
$$\frac{da}{dx} = \underbrace{\sigma'(z)}_{\text{local gradient at }\sigma} \cdot \underbrace{w}_{\text{local gradient at }z}.$$

## 1.2 Multivariate chain rule

If $z = f(x_1, \ldots, x_n)$ and each $x_i = g_i(t)$, then:
$$\frac{\partial z}{\partial t} = \sum_{i=1}^{n} \frac{\partial z}{\partial x_i}\cdot\frac{\partial x_i}{\partial t}.$$

This is the key formula for backpropagation: a node that *receives* input from multiple upstream sources accumulates gradient contributions from all of them.

**Vector form.** If $\mathbf{a} = f(\mathbf{z})$ where $\mathbf{z} = g(\mathbf{w})$, the chain rule gives:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \underbrace{\left(\frac{\partial \mathbf{z}}{\partial \mathbf{w}}\right)^{\!\top}}_{\text{Jacobian}^{\top}} \frac{\partial \mathcal{L}}{\partial \mathbf{z}}.$$"""))

C2.append(code(r"""# Visualise the chain rule: L(w) = cross_entropy(softmax(w*x + b), y)
# for a single scalar weight w, with x, b, y fixed

x_fixed = 2.0
b_fixed = -0.5
y_true  = np.array([1.0, 0.0])   # true class = 0

w_range = np.linspace(-3, 3, 300)

losses = []
for w in w_range:
    z = np.array([w * x_fixed + b_fixed, 0.0])   # 2-class logits
    losses.append(cross_entropy(softmax(z), y_true))
losses = np.array(losses)

# Exact gradient at w=1.0 via chain rule
w0 = 1.0
z0  = np.array([w0 * x_fixed + b_fixed, 0.0])
yh  = softmax(z0)
# d L / d z_0  =  y_hat_0 - y_true_0  (softmax + cross-entropy gradient)
dL_dz0 = yh[0] - y_true[0]
dz0_dw = x_fixed
dL_dw  = dL_dz0 * dz0_dw

L0 = cross_entropy(yh, y_true)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(w_range, losses, color='#3498db', lw=2.5, label=r'$\mathcal{L}(w)$')
ax.scatter([w0], [L0], color='#e74c3c', s=60, zorder=5)
ax.annotate('', xy=(w0 + 0.6, L0 + 0.6 * dL_dw),
            xytext=(w0, L0),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
ax.text(w0 + 0.65, L0 + 0.6 * dL_dw + 0.04,
        fr'slope $= {dL_dw:.3f}$', color='#e74c3c', fontsize=11)
ax.set_xlabel(r'weight $w$', fontsize=13)
ax.set_ylabel(r'loss $\mathcal{L}(w)$', fontsize=13)
ax.set_title(r'Loss as a function of a single weight; red arrow = gradient direction', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout(); plt.show()
print(f"Chain rule:  dL/dw = (dL/dz) * (dz/dw) = {dL_dz0:.4f} × {dz0_dw:.1f} = {dL_dw:.4f}")"""))

C2.append(md(r"""<a id="part2"></a>
# Part 2: Computational Graphs

## 2.1 Graphs as a bookkeeping device

Any differentiable expression can be represented as a **computational graph**: a directed acyclic graph (DAG) in which
- each **leaf node** holds an input or parameter ($x$, $w$, $b$, …),
- each **interior node** applies a single elementary operation (add, multiply, $\sigma$, …), and
- **edges** carry the result of each operation forward (and, during backprop, gradients backward).

**Example.** Consider the loss for a single ReLU unit:

$$\ell = \tfrac{1}{2}(a - y)^2, \quad\text{where}\quad a = \mathrm{ReLU}(z), \quad z = wx + b.$$

The graph has the following structure (read left to right):

$$x,\, w \;\longrightarrow\; [\times] \;\longrightarrow\; [+] \;\longleftarrow\; b \quad\longrightarrow\quad [\mathrm{ReLU}] \;\longrightarrow\; [\tfrac{1}{2}(\cdot - y)^2] \;\longleftarrow\; y$$

We will use concrete values: $x = 2,\; w = 0.5,\; b = -0.3,\; y = 1.0$.

## 2.2 Forward pass: computing values

The forward pass evaluates the graph left to right, storing every intermediate value.
Each stored value will be reused during backprop."""))

C2.append(code(r"""# ── Graph geometry ──────────────────────────────────────────────────────────
NODE_POS = {
    'x':    (0.0,  2.0),
    'w':    (0.0,  0.0),
    'mul':  (2.0,  1.0),
    'b':    (2.0, -1.0),
    'add':  (4.0,  1.0),
    'relu': (6.0,  1.0),
    'y':    (6.0, -1.0),
    'loss': (8.5,  1.0),
}
EDGES = [('x','mul'), ('w','mul'), ('mul','add'), ('b','add'),
         ('add','relu'), ('relu','loss'), ('y','loss')]

NODE_LABELS = {
    'x': r'$x$', 'w': r'$w$', 'b': r'$b$', 'y': r'$y$',
    'mul': r'$\times$', 'add': r'$+$',
    'relu': 'ReLU', 'loss': r'$\frac{1}{2}(a{-}y)^2$',
}
NODE_TYPE = {n: ('leaf' if n in ('x','w','b','y') else 'op') for n in NODE_POS}

# Forward-pass values
xv, wv, bv, yv = 2.0, 0.5, -0.3, 1.0
mulv = wv * xv         # = 1.0
addv = mulv + bv       # = 0.7  (this is z)
reluv = relu(addv)     # = 0.7  (this is a)
lossv = 0.5 * (reluv - yv)**2   # = 0.045

FWD_VALS = {
    'x': xv, 'w': wv, 'b': bv, 'y': yv,
    'mul': mulv, 'add': addv, 'relu': reluv, 'loss': lossv,
}
FWD_LABELS = {
    'x': f'x={xv}', 'w': f'w={wv}', 'b': f'b={bv}', 'y': f'y={yv}',
    'mul': f'wx={mulv}', 'add': f'z={addv}',
    'relu': f'a={reluv}', 'loss': f'ℓ={lossv:.3f}',
}

# Backward-pass gradients (∂ℓ/∂node)
dL_loss = 1.0
dL_relu = reluv - yv           # = -0.3   (∂ℓ/∂a)
dL_add  = dL_relu * d_relu(addv)   # = -0.3 * 1 = -0.3  (∂ℓ/∂z)
dL_mul  = dL_add * 1.0         # = -0.3   (∂ℓ/∂(wx), add has local grad 1)
dL_b    = dL_add * 1.0         # = -0.3   (∂ℓ/∂b)
dL_w    = dL_mul * xv          # = -0.6   (∂ℓ/∂w)
dL_x    = dL_mul * wv          # = -0.15  (∂ℓ/∂x)

BWD_GRADS = {
    'loss': dL_loss, 'relu': dL_relu, 'add': dL_add,
    'mul': dL_mul, 'b': dL_b, 'w': dL_w, 'x': dL_x, 'y': '—',
}

print("Forward-pass values:")
for k, v in FWD_VALS.items():
    print(f"  {NODE_LABELS[k]:>20s}  =  {v}")
print("\nBackward-pass gradients  ∂ℓ/∂(node):")
for k, v in BWD_GRADS.items():
    g = f'{v:.4f}' if isinstance(v, float) else v
    print(f"  ∂ℓ/∂({k:>5s})  =  {g}")"""))

C2.append(code(r"""def draw_graph(ax, reveal_fwd=None, reveal_bwd=None):
    # Draw the computational graph.
    # reveal_fwd: set of node names whose forward values are shown.
    # reveal_bwd: set of node names whose backward gradients are shown.
    ax.set_xlim(-0.8, 10.0)
    ax.set_ylim(-2.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    r = 0.55   # node radius

    # Edges
    for src, dst in EDGES:
        x0, y0 = NODE_POS[src]
        x1, y1 = NODE_POS[dst]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                   lw=1.4, connectionstyle='arc3,rad=0.0'))

    # Nodes
    for name, (x, y) in NODE_POS.items():
        is_leaf = NODE_TYPE[name] == 'leaf'
        if reveal_bwd and name in reveal_bwd and name != 'y':
            fc = '#fde8e8'   # red tint for backward-revealed nodes
            ec = '#c0392b'
        elif reveal_fwd and name in reveal_fwd:
            fc = '#e8f8f0'   # green tint for forward-revealed nodes
            ec = '#27ae60'
        else:
            fc = '#f5f5f5' if not is_leaf else '#eaf4ff'
            ec = '#999999'
        c = Circle((x, y), r, color=fc, ec=ec, lw=1.8, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, NODE_LABELS[name], ha='center', va='center',
                fontsize=10, zorder=4)

    # Forward values (green, above nodes)
    if reveal_fwd:
        for name in reveal_fwd:
            x, y = NODE_POS[name]
            ax.text(x, y + r + 0.22, FWD_LABELS[name],
                    ha='center', va='bottom', fontsize=9,
                    color='#27ae60', fontweight='bold')

    # Backward gradients (red, below nodes)
    if reveal_bwd:
        for name in reveal_bwd:
            if name == 'y':
                continue
            x, y = NODE_POS[name]
            g = BWD_GRADS[name]
            gstr = f'$\\partial\\ell/\\partial = {g:.3f}$' if isinstance(g, float) else ''
            ax.text(x, y - r - 0.22, gstr,
                    ha='center', va='top', fontsize=8.5,
                    color='#c0392b', fontweight='bold')

# Static forward-pass view
fig, ax = plt.subplots(figsize=(11, 4))
draw_graph(ax, reveal_fwd=set(FWD_VALS.keys()))
ax.set_title('Computational graph — forward pass values', fontsize=13)
plt.tight_layout(); plt.show()"""))

C2.append(md(r"""## 2.3 Interactive: step through the forward pass

Use the slider below to see values being computed left to right.
At each step, a new node is evaluated and its value stored."""))

C2.append(code(r"""FWD_ORDER = ['x', 'w', 'b', 'y', 'mul', 'add', 'relu', 'loss']

def _plot_fwd(step=0):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    revealed = set(FWD_ORDER[:step + 1]) if step >= 0 else set()
    draw_graph(ax, reveal_fwd=revealed)
    node_name = FWD_ORDER[min(step, len(FWD_ORDER) - 1)]
    ax.set_title(f'Forward pass  —  step {step}: computing {node_name}  '
                 f'(value = {FWD_LABELS[node_name]})', fontsize=12)
    plt.tight_layout(); plt.show()

interact(_plot_fwd, step=IntSlider(min=0, max=len(FWD_ORDER)-1,
                                    step=1, value=0, description='Step:'));"""))

C2.append(md(r"""<a id="part3"></a>
# Part 3: Backward Pass — Upstream × Local

## 3.1 The backpropagation rule

At each node $v$ in the graph, backpropagation computes:

$$\boxed{\frac{\partial\mathcal{L}}{\partial v} = \underbrace{\frac{\partial\mathcal{L}}{\partial u}}_{\text{upstream gradient}} \cdot \underbrace{\frac{\partial u}{\partial v}}_{\text{local gradient}}}$$

where $u$ is the node immediately downstream of $v$. If $v$ feeds into multiple nodes $u_1, u_2, \ldots$, the contributions are **summed**:
$$\frac{\partial\mathcal{L}}{\partial v} = \sum_i \frac{\partial\mathcal{L}}{\partial u_i} \cdot \frac{\partial u_i}{\partial v}.$$

The **local gradient** $\partial u / \partial v$ depends only on the operation at node $u$ — it can be computed analytically once and reused. The **upstream gradient** $\partial\mathcal{L}/\partial u$ is whatever was computed at the downstream node in the previous backward step.

## 3.2 Local gradients for our example

| Operation | Output $u$ | Input $v$ | Local gradient $\partial u/\partial v$ |
|-----------|-----------|-----------|---------------------------------------|
| $u = w \cdot x$ | mul | $w$ | $x$ |
| $u = w \cdot x$ | mul | $x$ | $w$ |
| $u = z_{\mathrm{mul}} + b$ | add | $z_{\mathrm{mul}}$ | $1$ |
| $u = z_{\mathrm{mul}} + b$ | add | $b$ | $1$ |
| $u = \mathrm{ReLU}(z)$ | relu | $z$ | $\mathbb{1}[z > 0]$ |
| $u = \tfrac{1}{2}(a-y)^2$ | loss | $a$ | $a - y$ |

These local gradients are **the same regardless of the current parameter values** — only the values of $x$, $w$, $b$ change at each training step."""))

C2.append(code(r"""BWD_ORDER = ['loss', 'relu', 'add', 'b', 'mul', 'w', 'x']

def _plot_bwd(step=0):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    revealed_fwd = set(FWD_VALS.keys())           # always show all forward values
    revealed_bwd = set(BWD_ORDER[:step + 1])
    draw_graph(ax, reveal_fwd=revealed_fwd, reveal_bwd=revealed_bwd)
    node_name = BWD_ORDER[min(step, len(BWD_ORDER) - 1)]
    g = BWD_GRADS[node_name]
    gstr = f'{g:.4f}' if isinstance(g, float) else str(g)
    ax.set_title(f'Backward pass  —  step {step}: ∂ℓ/∂({node_name}) = {gstr}', fontsize=12)
    plt.tight_layout(); plt.show()

interact(_plot_bwd, step=IntSlider(min=0, max=len(BWD_ORDER)-1,
                                    step=1, value=0, description='Step:'));"""))

C2.append(md(r"""> **Exercise 3.1.** *(Chain rule on the graph)*
>
> **(a)** Using the table of local gradients above, verify by hand (with $x=2, w=0.5, b=-0.3, y=1$) that:
> $$\frac{\partial\ell}{\partial w} = -0.6, \qquad \frac{\partial\ell}{\partial b} = -0.3, \qquad \frac{\partial\ell}{\partial x} = -0.15.$$
>
> **(b)** Suppose the unit uses **sigmoid** instead of ReLU. Redo the backward pass: what are $\partial\ell/\partial w$ and $\partial\ell/\partial b$ now? (Recall $\sigma'(z) = \sigma(z)(1-\sigma(z))$.)
>
> **(c)** In the backward step for the `mul` node, the local gradient with respect to $w$ is $x$ (the *value* of the other input). Explain why this means that weights receiving large-magnitude inputs tend to get large gradient updates."""))

C2.append(md(r"""<a id="part4"></a>
# Part 4: Backpropagation in a Feedforward Network

## 4.1 The $\boldsymbol{\delta}$ notation

For a network with $L$ layers we define the **error signal** at layer $l$ as:
$$\delta^{(l)} := \frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(l)}} \in \mathbb{R}^{n_l}.$$

This is the gradient of the loss with respect to the *pre-activations* of layer $l$.
Using the chain rule, the $\delta$'s satisfy a backward recurrence:

**Output layer** (softmax + cross-entropy, one-hot target $\mathbf{y}$):
$$\boxed{\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y}.}$$
This elegant formula is the combined gradient of softmax and cross-entropy.

**Hidden layer** $l < L$ (element-wise activation $\sigma$):
$$\boxed{\delta^{(l)} = \left(W^{(l+1)\top}\,\delta^{(l+1)}\right) \odot \sigma'\!\left(\mathbf{z}^{(l)}\right).}$$

Here $\odot$ denotes element-wise multiplication. The two factors reflect:
- $W^{(l+1)\top}\,\delta^{(l+1)}$: upstream gradient arriving from layer $l+1$, "broadcast back" through the weight matrix (the local gradient of the linear map);
- $\sigma'(\mathbf{z}^{(l)})$: local gradient of the activation function.

## 4.2 Parameter gradients

Once the $\delta^{(l)}$ are in hand, the weight and bias gradients are:
$$\boxed{\frac{\partial\mathcal{L}}{\partial W^{(l)}} = \delta^{(l)}\!\left(\mathbf{a}^{(l-1)}\right)^{\!\top}, \qquad \frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(l)}} = \delta^{(l)}.}$$

These formulas follow immediately from the chain rule applied to the affine map $\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$.

**Mnemonic:** $\nabla_{W^{(l)}}\mathcal{L} = \underbrace{\delta^{(l)}}_{\text{what layer }l\text{ should have been}} \underbrace{(\mathbf{a}^{(l-1)})^{\top}}_{\text{what it received}}$."""))

C2.append(md(r"""## 4.3 Derivation of the output-layer formula

The cross-entropy loss for a one-hot target $\mathbf{y}$ is $\mathcal{L} = -\sum_k y_k \log \hat{y}_k$.
With $\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{z}^{(L)})$, a direct computation gives:

$$\frac{\partial\mathcal{L}}{\partial z_k^{(L)}} = \hat{y}_k - y_k, \qquad k = 1, \ldots, K.$$

In vector form: $\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$.

**Why is this so clean?** The softmax and cross-entropy gradients "cancel" each other's complexity. The Jacobian of softmax contains cross-terms $\partial\hat{y}_k/\partial z_j = -\hat{y}_k\hat{y}_j$ for $k \ne j$, but when composed with the cross-entropy gradient, all cross-terms collapse and the result is the simple residual $\hat{\mathbf{y}} - \mathbf{y}$.

> **Exercise 4.1.** *(Deriving $\delta^{(L)}$)*
>
> Derive $\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$ from scratch.
>
> **(a)** Write $\mathcal{L} = -\sum_k y_k \log \hat{y}_k$ and $\hat{y}_k = e^{z_k}/\sum_j e^{z_j}$.
> Compute $\partial\mathcal{L}/\partial z_k$ using the quotient rule for softmax.
>
> *Hint:* split into two cases: $j = k$ (diagonal) and $j \ne k$ (off-diagonal).
>
> **(b)** Now substitute back into $\partial\mathcal{L}/\partial z_k$ and simplify using $\sum_j y_j = 1$."""))

C2.append(md(r"""## 4.4 Derivation of the hidden-layer recurrence

For layer $l < L$, we apply the chain rule through layer $l+1$:

$$\delta_k^{(l)} = \frac{\partial\mathcal{L}}{\partial z_k^{(l)}} = \sum_{j=1}^{n_{l+1}} \frac{\partial\mathcal{L}}{\partial z_j^{(l+1)}} \cdot \frac{\partial z_j^{(l+1)}}{\partial z_k^{(l)}}.$$

Now, $z_j^{(l+1)} = \sum_k W^{(l+1)}_{jk}\,a_k^{(l)} + b_j^{(l+1)}$ and $a_k^{(l)} = \sigma(z_k^{(l)})$, so:

$$\frac{\partial z_j^{(l+1)}}{\partial z_k^{(l)}} = W^{(l+1)}_{jk}\,\sigma'\!\left(z_k^{(l)}\right).$$

Substituting:
$$\delta_k^{(l)} = \sigma'\!\left(z_k^{(l)}\right) \sum_{j} W^{(l+1)}_{jk}\,\delta_j^{(l+1)} = \sigma'\!\left(z_k^{(l)}\right)\cdot\left[W^{(l+1)\top}\delta^{(l+1)}\right]_k.$$

In vector notation: $\delta^{(l)} = \bigl(W^{(l+1)\top}\delta^{(l+1)}\bigr) \odot \sigma'(\mathbf{z}^{(l)})$.

> **Exercise 4.2.** *(Hidden-layer recurrence)*
>
> **(a)** Fill in the full derivation of $\nabla_{W^{(l)}}\mathcal{L} = \delta^{(l)}(\mathbf{a}^{(l-1)})^\top$ starting from the chain rule applied to the affine map $\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$.
>
> **(b)** For ReLU activations, what is $\sigma'(\mathbf{z}^{(l)})$? What does it mean for backpropagation when many units have $z_k^{(l)} < 0$?"""))

C2.append(md(r"""<a id="part5"></a>
# Part 5: NumPy Implementation and Gradient Check

We implement the full forward + backward pass on a $(3 \to 4 \to 2)$ network with ReLU hidden units and softmax+cross-entropy output — the same network traced in [`fwdpass_units.ipynb`](fwdpass_units.ipynb)."""))

C2.append(code(r"""np.set_printoptions(precision=6, suppress=True)
rng5 = np.random.default_rng(0)

# ── Network parameters ───────────────────────────────────────────────────────
W1 = rng5.standard_normal((4, 3)) * 0.5
b1 = np.zeros(4)
W2 = rng5.standard_normal((2, 4)) * 0.5
b2 = np.zeros(2)

# ── A single training example ────────────────────────────────────────────────
x     = np.array([0.5, -1.2, 0.8])
y_one_hot = np.array([1.0, 0.0])   # true class = 0

# ════════════════════════════════════════════════════════════════════════════
# FORWARD PASS
# ════════════════════════════════════════════════════════════════════════════
a0 = x

z1 = W1 @ a0 + b1
a1 = relu(z1)              # ReLU hidden layer

z2 = W2 @ a1 + b2
y_hat = softmax(z2)        # softmax output

loss = cross_entropy(y_hat, y_one_hot)

print("=" * 54)
print("FORWARD PASS")
print("=" * 54)
print(f"  z⁽¹⁾     = {z1}")
print(f"  a⁽¹⁾     = {a1}   ({(z1<0).sum()} units zeroed)")
print(f"  z⁽²⁾     = {z2}")
print(f"  ŷ        = {y_hat}   (sum={y_hat.sum():.6f})")
print(f"  loss     = {loss:.6f}")

# ════════════════════════════════════════════════════════════════════════════
# BACKWARD PASS
# ════════════════════════════════════════════════════════════════════════════
# Output layer: δ⁽²⁾ = ŷ - y
delta2 = y_hat - y_one_hot

# Gradients for layer 2
dW2 = np.outer(delta2, a1)     # δ⁽²⁾ (a⁽¹⁾)ᵀ
db2 = delta2

# Hidden layer: δ⁽¹⁾ = (W⁽²⁾ᵀ δ⁽²⁾) ⊙ σ'(z⁽¹⁾)
delta1 = (W2.T @ delta2) * d_relu(z1)

# Gradients for layer 1
dW1 = np.outer(delta1, a0)     # δ⁽¹⁾ (a⁽⁰⁾)ᵀ  = δ⁽¹⁾ xᵀ
db1 = delta1

print("\n" + "=" * 54)
print("BACKWARD PASS")
print("=" * 54)
print(f"  δ⁽²⁾     = {delta2}   (= ŷ - y)")
print(f"  δ⁽¹⁾     = {delta1}")
print(f"  ∇_W⁽²⁾ ℒ shape: {dW2.shape}    max|grad| = {np.abs(dW2).max():.4f}")
print(f"  ∇_W⁽¹⁾ ℒ shape: {dW1.shape}   max|grad| = {np.abs(dW1).max():.4f}")"""))

C2.append(md(r"""## 5.1 Numerical gradient check

The only reliable way to verify a backprop implementation is the **gradient check**: compare the analytic gradient to a finite-difference estimate.

For parameter $\theta_i$, the two-sided finite difference approximation is:
$$\frac{\partial\mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\boldsymbol{\theta} + \varepsilon\mathbf{e}_i) - \mathcal{L}(\boldsymbol{\theta} - \varepsilon\mathbf{e}_i)}{2\varepsilon}.$$

For well-implemented backprop, the **relative error**
$$\frac{\|\nabla_{\mathrm{analytic}} - \nabla_{\mathrm{numerical}}\|}{\|\nabla_{\mathrm{analytic}}\| + \|\nabla_{\mathrm{numerical}}\|}$$
should be $\ll 10^{-5}$."""))

C2.append(code(r"""def forward_loss(W1, b1, W2, b2, x, y):
    # Full forward pass, returns scalar loss.
    a1 = relu(W1 @ x + b1)
    y_hat = softmax(W2 @ a1 + b2)
    return cross_entropy(y_hat, y)

def numerical_grad(param, grad_fn, eps=1e-5):
    # Two-sided finite-difference gradient for a flat parameter vector.
    p = param.ravel().copy()
    num = np.zeros_like(p)
    for i in range(len(p)):
        p[i] += eps; lp = grad_fn(p)
        p[i] -= 2*eps; lm = grad_fn(p)
        p[i] += eps
        num[i] = (lp - lm) / (2 * eps)
    return num.reshape(param.shape)

def rel_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)

# Numerical gradients
num_dW2 = numerical_grad(W2, lambda p: forward_loss(W1, b1, p.reshape(W2.shape), b2, x, y_one_hot))
num_db2 = numerical_grad(b2, lambda p: forward_loss(W1, b1, W2, p, x, y_one_hot))
num_dW1 = numerical_grad(W1, lambda p: forward_loss(p.reshape(W1.shape), b1, W2, b2, x, y_one_hot))
num_db1 = numerical_grad(b1, lambda p: forward_loss(W1, p, W2, b2, x, y_one_hot))

print("Gradient check  (relative error should be < 1e-5)")
print(f"  W2:  {rel_error(dW2, num_dW2):.2e}")
print(f"  b2:  {rel_error(db2, num_db2):.2e}")
print(f"  W1:  {rel_error(dW1, num_dW1):.2e}")
print(f"  b1:  {rel_error(db1, num_db1):.2e}")"""))

C2.append(md(r"""## 5.2 Connection to SGD

Once we have $\nabla_{W^{(l)}}\mathcal{L}$ and $\nabla_{\mathbf{b}^{(l)}}\mathcal{L}$, a gradient descent step is simply:
$$W^{(l)} \;\longleftarrow\; W^{(l)} - \eta\,\nabla_{W^{(l)}}\mathcal{L}, \qquad \mathbf{b}^{(l)} \;\longleftarrow\; \mathbf{b}^{(l)} - \eta\,\nabla_{\mathbf{b}^{(l)}}\mathcal{L},$$
where $\eta > 0$ is the learning rate. This is exactly the update used in [`optimization_sgd.ipynb`](optimization_sgd.ipynb).

In **mini-batch SGD**, the gradients above are averaged over a batch of $B$ examples, and the update is applied after each batch. Keras handles all of this internally; our NumPy implementation here exposes the machinery underneath.

> **Exercise 5.1.** *(Extending to a batch)*
>
> **(a)** Modify the forward/backward code above to handle a batch of $B$ inputs, stored as a matrix $X \in \mathbb{R}^{B \times n_0}$ (one row per example). The gradients should be averaged over the batch.
>
> *Hint:* The weight gradient formula $\nabla_{W^{(l)}}\mathcal{L} = \delta^{(l)}(\mathbf{a}^{(l-1)})^\top$ for a single example generalises to $\frac{1}{B}\Delta^{(l)\top} A^{(l-1)}$ where the columns of $\Delta^{(l)}$ are the individual $\delta^{(l)}$ vectors.
>
> **(b)** Implement one step of gradient descent using your batch backprop, and check that the loss decreases.

> **Exercise 5.2.** *(Modifying the network)*
>
> The code currently uses ReLU hidden units. Adapt the backward pass to use **sigmoid** instead, and re-run the gradient check.
>
> **(a)** What one line of the backward pass needs to change?
>
> **(b)** Does the gradient check still pass? Compare the magnitudes of $\delta^{(1)}$ between the ReLU and sigmoid cases with the same initialisation. What do you observe?"""))

C2.append(md(r"""---
# Summary

| Concept | Key formula |
|---------|-------------|
| **Chain rule (scalar)** | $\dfrac{da}{dx} = \dfrac{da}{dz}\cdot\dfrac{dz}{dx}$ |
| **Backprop rule** | $\dfrac{\partial\mathcal{L}}{\partial v} = \dfrac{\partial\mathcal{L}}{\partial u} \cdot \dfrac{\partial u}{\partial v}$ (upstream × local) |
| **Output error** | $\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$ (softmax + cross-entropy) |
| **Hidden error** | $\delta^{(l)} = \bigl(W^{(l+1)\top}\delta^{(l+1)}\bigr)\odot\sigma'(\mathbf{z}^{(l)})$ |
| **Weight gradient** | $\nabla_{W^{(l)}}\mathcal{L} = \delta^{(l)}\bigl(\mathbf{a}^{(l-1)}\bigr)^{\!\top}$ |
| **Bias gradient** | $\nabla_{\mathbf{b}^{(l)}}\mathcal{L} = \delta^{(l)}$ |
| **Cost of backprop** | $O(P)$ operations — same order as a forward pass |

**Key insight:** backpropagation is not a new algorithm — it is simply the chain rule applied systematically to a computational graph, traversed in reverse topological order. Modern deep-learning frameworks (JAX, PyTorch, TensorFlow) implement this automatically via **automatic differentiation**."""))

# ════════════════════════════════════════════════════════════════════════════
# SAVE BOTH NOTEBOOKS
# ════════════════════════════════════════════════════════════════════════════

out1 = pathlib.Path('~/repos/MathAIspring2026UTSA/code/fwdpass_units.ipynb')
out2 = pathlib.Path('~/repos/MathAIspring2026UTSA/code/backpropagation.ipynb')

out1.write_text(json.dumps(notebook(C1), indent=1, ensure_ascii=False))
out2.write_text(json.dumps(notebook(C2), indent=1, ensure_ascii=False))

print(f"Written: {out1}  ({len(C1)} cells)")
print(f"Written: {out2}  ({len(C2)} cells)")
