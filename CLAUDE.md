# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository powers the **MAT 4953/MAT 6973 Mathematical Foundations of AI** course website at UTSA (Spring 2026), taught by Dr. Eduardo Dueñez. It is a Jekyll static site deployed to GitHub Pages, with supplementary Python code and Jupyter notebooks for AI/ML topics.

## Local Development

```sh
bundle install
bundle exec jekyll serve
```

Site runs at http://localhost:4000. Never edit files in `_site/` — it is auto-generated.

## Running Jupyter Notebooks Locally

```sh
source .venv/bin/activate      # activate the project virtualenv
pip install -r requirements.txt # install dependencies (first time / after updates)
jupyter notebook --notebook-dir=code
```

This opens the Jupyter dashboard scoped to the `code/` directory. Notebooks can also be run directly in VS Code with the Jupyter extension — select the `.venv` kernel when prompted.

## Architecture

**Jekyll Collections** (under `collections/`):
- `_modules/` — Weekly learning modules (`week-01.md` through `week-09.md`)
- `_assignments/` — Course assignments
- `_announcements/` — Course announcements
- `_refdocs/` — Reference materials (books, frameworks, LaTeX guides, writing guidelines)
- `_staffers/` — Course staff profiles
- `_schedules/` — Schedule/calendar displays

**Templates & Styling**:
- `_layouts/` — Page templates (`module.html`, `schedule.html`, etc.)
- `_includes/` — Shared HTML snippets; update `nav_footer_custom.html` to change navigation
- `_sass/custom/` — Custom SCSS; import new files via `custom.scss`

**Code** (`code/`): Jupyter notebooks and Python scripts for course examples. Current notebooks:

| File | Topic | Notes |
|------|-------|-------|
| `intro_feedforward_mnist.ipynb` | MNIST feedforward network (Keras/JAX) | First-week bird's-eye overview; less polished than later notebooks |
| `numpy_tutorial_svd.ipynb` | NumPy & SVD tutorial | |
| `bias_variance_regression.py` | Bias-variance interactive tool | ipywidgets slider + button |
| `bias_variance_double_descent.ipynb` | Capacity / U-curve / double descent visualization | |
| `optimization_sgd.ipynb` | SGD, momentum, Adam | Polished lesson; heavy widget use; reference for house style |
| `svm_lesson.ipynb` | SVMs (geometry → kernels) | Polished lesson; heaviest math; reference for house style |
| `fwdpass_units.ipynb` | Feedforward networks: forward pass & activation functions | Sequel to `intro_feedforward_mnist`; introduces units, ReLU/GeLU/softmax |
| `backpropagation.ipynb` | Backpropagation algorithm | Requires `fwdpass_units`; chain rule, computational graphs, δ recurrence, gradient check |
| `dnn_architectures_overview.ipynb` | DNN architecture families (MLP, CNN, RNN, Transformer, AE) | Comparative survey with toy examples |
| `generative_models.ipynb` | Generative models (PCA → VAE → LLMs) | ELBO, reparameterization trick, autoregressive generation, temperature |
| `sequence_memory_models.ipynb` | Memory in sequence models (LSTM → KV cache → Mamba) | Gates, KV cache, SSM recurrence–convolution duality |
| `diffusion_models.ipynb` | Score-based generative models and DDPM | Forward/reverse diffusion, score functions, Langevin sampling, denoising = score matching |
| `reinforcement_learning.ipynb` | RL: MDPs, Bellman, policy gradients, RLHF | Gridworld, value iteration, REINFORCE, Bradley–Terry reward model, KL-penalised objective |
| `geometric_deep_learning.ipynb` | Symmetry, groups, GNNs, WL test | Equivariance, group actions, message passing, GCN, WL expressivity, E(3) for 3D science |

**Notebook generation scripts** (in `code/tmp/`):
- `gen_ff_notebooks.py` → generated `fwdpass_units.ipynb` and `backpropagation.ipynb`; references `intro_feedforward_mnist.ipynb` and `optimization_sgd.ipynb` in cell content
- `gen_advanced_notebooks.py` → generated `diffusion_models.ipynb`, `reinforcement_learning.ipynb`, `geometric_deep_learning.ipynb`

Keep these updated if notebooks are regenerated from scratch.

**Python dependencies**: `requirements.txt` (numpy, matplotlib, scikit-learn, scipy, jax, keras, jupyter, ipywidgets). `geometric_deep_learning.ipynb` additionally requires `networkx` (auto-installed on first run) and `scipy.spatial.transform` (included in scipy).

## Content Conventions

- All content is Markdown with YAML front matter. Use Liquid templating for dynamic content.
- Enable MathJax per-page with `math: mathjax3` in front matter.
- Collection files follow the naming pattern of their collection (e.g., `_modules/week-01.md`).
- Do not commit `_site/` or backup files (`*.md~`, `#*`).
- No custom build scripts — only standard Jekyll.

## Notebook House Style

`svm_lesson.ipynb` and `optimization_sgd.ipynb` are the canonical style references. Key conventions:

- **Header**: H1 title, H2 subtitle, course line (`**MAT 4953/6973 — Mathematical Foundations of AI** (Spring 2026, UTSA)`), Colab badge, `---`
- **Sections**: numbered Parts at H1, numbered subsections at H2 (e.g., `## 2.1`)
- **Math**: `$$...$$` for display equations; box key results with `\boxed{}`; use `\begin{aligned}` for multi-line
- **Exercises**: blockquote format — `> **Exercise N.M.** *(title)*` with lettered sub-parts `**(a)**`, `**(b)**`, …
- **Experiments**: inline `**Experiment:**` prompts with bullet "what if" questions
- **Interactivity**: `ipywidgets.interact` with sliders/dropdowns; mark tunable parameters with `# @param` comments
- **Figures**: blue `#3498db` / red `#e74c3c` for two-class data; gold for highlights; always `edgecolors='k', lw=0.5` on scatter; `plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'figure.dpi': 100})`
- **Flow**: theory cell → code/visualization cell → discussion or exercise; avoid docstrings inside `code()` string literals when generating notebooks programmatically (use `#` comments instead)
