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

**Code** (`code/`): Jupyter notebooks and Python scripts for course examples (MNIST feedforward networks, NumPy/SVD tutorial, bias-variance visualizations with ipywidgets).

**Python dependencies**: `requirements.txt` (numpy, matplotlib, scikit-learn, scipy, jax, keras, jupyter, ipywidgets)

## Content Conventions

- All content is Markdown with YAML front matter. Use Liquid templating for dynamic content.
- Enable MathJax per-page with `math: mathjax3` in front matter.
- Collection files follow the naming pattern of their collection (e.g., `_modules/week-01.md`).
- Do not commit `_site/` or backup files (`*.md~`, `#*`).
- No custom build scripts — only standard Jekyll.
