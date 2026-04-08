---
layout: page
title: Final presentation & paper
description: >-
    Guidelines for final in-class presentations and final paper
nav_order: 0
---

# Final presentation and final paper
{:.no_toc}

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Schedule

Final presentations will be on **Monday 11 May, 5:00–6:50 PM** (date/time of scheduled final exam).
**Final papers are also due** at that time.
Papers must be submitted in **hardcopy**.

<!-- [**Click to sign up for a presentation slot**{: .label .label-red }](#) -->

## Choice of topic

Your final project (presentation and paper) should address a topic in the *mathematical foundations of AI/Deep Learning*.

- The topic may be a **continuation or deepening** of your midterm presentation topic, or an entirely **new topic**.
  If continuing your midterm topic, the final project must represent a *substantial extension* — not merely a repetition of the same material.
- The topic should demonstrate *significant mathematical content* commensurate with your standing as an upper-level undergraduate or graduate student.
- Applied topics are acceptable provided the mathematical treatment is substantive (e.g., rigorous analysis of an algorithm, convergence proofs, derivations of key formulas, complexity analysis).
- Purely expository surveys are acceptable if they synthesize material from multiple sources and present it with mathematical depth and coherence.

It is **strongly recommended** to discuss your choice of topic with the instructor at least two weeks in advance of the presentation date.

## Final presentation (15% of final grade)

### Format

The presentation will be oral, delivered during the final exam time slot.

- You are encouraged to use slides, diagrams, figures, short demonstrations, or a Colab notebook to illustrate key concepts.
  Visual aids are especially important for topics involving architectures, algorithms, or experimental results.
- Your presentation should last **8 to 10 minutes**.
  - The 10-minute limit is **absolute**; you should plan and aim for an *8-minute* presentation.
    There is *no credit to gain* by exceeding 8 minutes.
  - Any presentation lasting over 10 minutes will be subject to a **10% credit deduction** for *each additional minute* (or fraction) exceeding the hard 10-minute limit.
- At the end of your presentation, there will be a **2-minute** interval for questions (Q & A) from the audience.
  - If your presentation exceeds 10 minutes, the excess will *decrease* your allotted Q & A time.
  - If no other student asks a question, there will be an automatic 10% deduction.
    (In that case, the instructor will ask at least one question.)
    Give a *clear and engaging* presentation that invites audience questions!

### Content expectations

A strong presentation should:

1. **Motivate**: State the topic and explain its interest or importance to AI/Deep Learning.
2. **Technical content**: It should involve mathematics at the level of your formation (undergraduate/graduate).
   Although you are not required to present a proof (although you can), your talk should include some mathematical result(s) or some theorem(s):
   this is the technical soul of your presentation!
3. **Illustrate/exemplify**: Include at least one concrete example, figure, or computational demonstration related to the mathematical results.
4. **Summarize and invite further study**: Summarize at the end, mentioning some direction(s) in which the topic extends beyond the presentation (perhaps referring to material in your final paper but not included in the presentation). 

You are also required to design a handout, at most **2 pages** in length, meant to complement your presentation summarizing key results and/or serving as a mini-lesson (perhaps including exercises or hands-on activities).

### Grading Criteria (final presentation)

| Component | Weight |
|-----------|--------|
| Choice of topic (interesting, relevant, suitable sophistication) | 15% |
| Technical content (correctness, coherence, completeness, depth) | 35% |
| Delivery (format, polish, flow, audience engagement, time management) | 30% |
| Visual aids and examples (quality, clarity, relevance) | 20% |

## Final paper

### Format and length

- **Sources**: The paper must be based on at least **2 primary sources** plus **1 secondary source**.
  Primary sources *must* be in the published/research literature or book chapters (*not* from the textbook for this course). 
  (Of course, such papers or book chapters may still be available electronically.)
  References to “web-only” resources are discouraged, although they are acceptable as *secondary* sources if they are authoritative (e.g., Wikipedia pages or preprints).
- The paper must be **typeset using LaTeX** single-spaced at 12 point font size with 1in margins.
  The title/header must not occupy more than 20% of the first page.
  A [LaTeX template]({{ site.baseurl }}/code/MathAI_HW_template.tex) is available;
  you are encouraged to base your final paper on it.
- **Length**: A *minimum* of 5 pages, and a “soft maximum” limit of 6 pages of content proper (it may be exceeded, but not excessively), plus references and appendices (possibly in excess of 6 pages).
  Figures, diagrams, and tables should be included where they aid understanding;
  however, any such figures/diagrams/tables in excess of 1-page total length will not count towards the 5-page minimum length.
- The paper must include a **bibliography** with properly formatted references (use BibTeX).

### Structure

Your paper should follow a clear structure. A recommended outline:

1. **Title, Author and Date.**
2. **Abstract** (≲100 words): 
   A concise summary of the topic.
3. **Introduction**: 
   Full statement of the topic, motivation and context within AI/deep learning.
4. **Background/Preliminaries**:
   (If necessary.) Any necessary prelude to the main content.
5. **Main content** (one or more sections): 
   The core content, including mathematical results, theorems, proofs, derivations.
   The rigorous analysis of algorithms (if applicable) is also mathematical content proper.
6. **Examples or experiments** (if applicable):
   Illustrations, figures, computational results, or code demonstrations;
   each *must* have a suitably descriptive caption explaining.
7. **Conclusion**: 
   Summary of main results, significance, and potential directions for further study.
8. **References.**
   List of references cited.
   Do *not* include “ghost” references to material/sources that are (possibly) related to the paper’s topic, but not used or directly cited.

### Content expectations

- The paper need not contain original research, but it must contain **original exposition**:
  You are expected to *synthesize, explain and present* the material from *multiple sources* in your *own words* and with your *own organization*.
  Do not write a “patchwork” of heterogeneous sources:
  use *your own* academic voice and writing.
- **Mathematical rigor** is essential: state assumptions clearly, define all notation, and provide proofs or detailed derivations where appropriate.
- If your paper includes **code or experiments**, describe the methodology clearly and present results with appropriate figures or tables.
  Shared all your code (e.g., via a linked Colab notebook or GitHub repository).
- Per the [syllabus]({{ site.baseurl }}{% link syllabus.md %}), AI is **allowed** as an aid in coding and as an aid to writing (grammar, style, spelling, organization, LaTeX formatting), but **not** to generate content proper (e.g., proofs, solutions, or expository text).
  **All uses of AI must be properly acknowledged.**
  Any figures, diagrams or code generated with the assistance of AI must be explicitly so acknowledged in their caption.

Direct copying from sources (apart from a very few and properly quoted sentences or short paragraphs) is **plagiarism** subject to sanctions such as disciplinary probation, withholding transcripts or expulsion, depending on gravity and prior offenses per section 203 of the UTSA Student Code of Conduct.
**This extends and applies to diagrams or tables copied verbatim.**
Direct reproduction requires permission from the original author(s).

### Sharing digital assets

*Before the deadline (Monday 11 May 5:00pm)* provide and share the following Digital Assets:
- Presentation plus handout 
  (`.tex`, `.pdf`, plus any other files needed for compilation);
- Final paper 
  (`.tex`, `.pdf`, also `.bib` if using RefTeX, plus any other figures or files needed for compilation);
- Python code (if applicable —provide link to a *shared* Colab notebook or *public* Github repository);
- Etc. (any other assets you want to share, e.g., additional web links).

The procedure is as follows:
I have created a [**shared Google Drive folder**](https://drive.google.com/drive/folders/109Xl4t71CCNnOUshZ2eCLcvnbUoRs9Qu?usp=sharing){:target="_blank"}.
(Currently, nobody has access yet.)
I will call this shared folder the **Digital Assets (DA) container**;
it will be accessible (*readable*) by every student in the class
(each student will be granted *write* rights to their own sub-folder **“DA subfolder”**).

Each DA subfolder will have a template document *“Web_and_Colab_links”* where you are meant to keep the list of *Colab links* and other *web links/URLs* to the location(s) where you host shared digital assets.
(For instance, you can include a link to a public Github repository where you keep some or all of your digital assets.)
**Make sure all shared web links are public so everyone can access them**.

If you do not wish to share web links publicly, you can upload all assets directly to your DA subfolder; you can even upload verbatim copies of your Colab notebooks!

Please do **not** attempt to modify or delete other students’ subfolders.

### Grading (final paper — 15% of course grade)

| Component | Weight |
|-----------|--------|
| Choice of topic and scope (depth, relevance, ambition) | 10% |
| Technical content (correctness, rigor, completeness) | 40% |
| Exposition and writing quality (clarity, organization, flow) | 25% |
| Presentation of figures, examples, and references | 25% |

## General policies

- The same [guidelines on the use of generative AI]({{ site.baseurl }}{% link syllabus.md %}#guidelines-for-the-use-of-generative-artificial-intelligence) from the syllabus apply to both the final presentation and paper.
- The final paper is due in **hardcopy** at the start of the “final exam” session on Monday 11 May at 5:00 pm.
  Late submissions will not be accepted.
