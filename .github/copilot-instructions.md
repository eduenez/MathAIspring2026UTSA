
# Copilot Instructions for MathAIspring2026UTSA

## Project Overview
This repository powers the "MAT 4953/MAT 6973 Mathematical Foundations of AI" course at UTSA. It is a Jekyll-based static site for course content, announcements, schedules, and reference materials, with example code and Jupyter notebooks for AI/ML topics.

## Architecture & Structure
- **Jekyll Static Site**: Content is organized via markdown and Liquid templates. The site is built with Jekyll and outputs to `_site/` (never edit this folder).
- **Collections**: Custom Jekyll collections for announcements, assignments, modules, reference docs, schedules, and staffers. Each collection is a folder under `collections/` (e.g., `_modules/`, `_announcements/`).
- **Templates & Includes**: Page layouts in `_layouts/` (e.g., `module.html`, `schedule.html`). Shared HTML snippets in `_includes/` (e.g., `nav_footer_custom.html`).
- **Styling**: Custom SCSS in `_sass/custom/`, imported via `custom.scss`.
- **Assets**: Images and custom JS/CSS in `assets/`.
- **Code & Notebooks**: Example code and Jupyter notebooks (e.g., `mnist_ff.ipynb`) in `code/`.

## Developer Workflows
- **Local Build/Preview**:
  ```sh
  bundle install
  bundle exec jekyll serve
  ```
  Site runs at http://localhost:4000.
- **Content Addition**: Add new modules, announcements, etc. as markdown files in the appropriate collection. Use YAML front matter for metadata.
- **Styling**: Add/modify SCSS in `_sass/custom/`. Import new files in `custom.scss`.
- **Navigation**: Update navigation in `_includes/nav_footer_custom.html`.
- **Code/Notebooks**: Place new code or Jupyter notebooks in `code/`.
- **Do Not Edit**: Never edit files in `_site/` (auto-generated), or commit changes to it.

## Project Conventions
- **Markdown-First**: All content is markdown with YAML front matter. Use Liquid for dynamic content.
- **Naming**: Collection files start with an underscore and are grouped by type (e.g., `_modules/week-01.md`).
- **No Custom Build Scripts**: Only standard Jekyll build process is used.
- **Version Control**: Do not commit `_site/` or backup files (e.g., `*.md~`).

## Integration Points & Dependencies
- **Jekyll Plugins**: Add to `Gemfile` and configure in `_config.yml` if needed.
- **MathJax**: Math rendering via `math: mathjax3` in front matter.
- **Assets**: Reference images and JS/CSS from `assets/`.
- **Notebooks**: Place and update Jupyter notebooks in `code/`.

## Examples & Patterns
- **Add a Module**: Create a markdown file in `_modules/` with YAML front matter (see `week-01.md`).
- **Update Navigation**: Edit `_includes/nav_footer_custom.html`.
- **Add Style**: Edit/add SCSS in `_sass/custom/` and import in `custom.scss`.
- **Reference Docs**: Add markdown or HTML to `collections/_refdocs/`.

## References
- See `README.md` for course description and high-level overview.
- See `_config.yml` for Jekyll configuration and enabled plugins.
- See `Gemfile` for Ruby dependencies.

---
For project-specific questions, consult the course instructor or Jekyll documentation. Keep all content and code organized by collection and follow the markdown/Liquid conventions shown in this repo.
