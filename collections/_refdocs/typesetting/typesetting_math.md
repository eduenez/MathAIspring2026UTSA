---
layout: page
title: Typesetting Mathematics
nav_exclude: true
output: true
description: >-
    Guide to writing mathematics
math: mathjax3
---

# Guide to typesetting mathematics

*Disclaimer:* The instructor does not have the means to provide technical support.
You are encouraged to reach out to classmates who may already be familiar with TeX and its use.
(A tea or coffee break often helps when one is stuck.)

Word processors like MS Word, Pages, and OpenOffice Writer are general-purpose tools poorly suited for writing documents containing many mathematical formulas.
There are much better, specialized professional tools available to write math.

## TeX and LaTeX

The 800-pound gorilla in the world of mathematical typesetting is the *TeX* system (pronounced: *“tek”* as in *“tech-nic-al”*—with apologies to Texans, never *“teks”*\!).
*TeX is to math writing as Word is to word processing:*
It is the default program with the largest “market share”, considered the standard in its class.
(Moreover, the TeX software itself is free.)

The TeX system is the professional typesetting tool used by more mathematicians and companies publishing technical manuscripts with a lot of math (also, by lots of computer scientists, chemists, and even musicians who need to typeset sheet music).
Nowadays, rather than using TeX proper, the vast majority of mathematicians use *LaTeX*, which is a more modern extension of TeX (still relying on the TeX engine as a foundation);
the term “TeX” is understood to mean “LaTeX” in practice, by default.

### Writing TeX (and LaTeX)

Writing TeX is a bit like writing a simple computer program.
For instance, to create a document displaying the formula
\\(c = a^2+b^2,\\)
you actually need to enter (i.e., type literally) the text
`c = \sqrt{a^2 + b^2}`
on an (otherwise regular text) file with extension `.tex`.
(Note: *LaTeX* files still have the extension `.tex` usually.)
After typing text directly on the `.tex` file, a second step (“compilation”) is necessary in order to create a PDF file that actually displays the formula.
This will sound familiar if you have written computer code that needs to be compiled (i.e., processed/packaged) before it can be actually used/deployed.
Writing TeX is very similar to composing a hypertext (such as HTML) or markup (such as Markdown/MD) document which needs to be pre-processed before it becomes human-readable standard text.

You’ll find it’s pretty easy to learn LaTeX after picking up the basics.

## Recommendations

### The “easy way”: Write your LaTeX documents using an interactive web interface

- A popular online system is *Overleaf* [https://www.overleaf.com](https://www.overleaf.com/).
Students (particularly those having limited experience with programming) generally love Overleaf’s simplicity.
In practice, “free” Overleaf is fine for single-author projects.
In my opinion, paid Overleaf for multi-author collaborations is not reasonably priced—even for students enjoying “discounts”.
Although easiest to use, I dislike the limitations of free Overleaf accounts, so I never use it by choice.

- *CoCalc* [https://www.cocalc.com/](https://www.cocalc.com/) is a comprehensive cloud computing platform which I recommend (and personally use).
The free tier of CoCalc allows collaborative LaTeX editing among any number of authors.
	Basic paid tiers cost no more than Overleaf but offer an order of magnitude more functionality, e.g., Virtual Machines running a Linux/Unix shell atop Python, R, Julia among other scientific computing platforms.

### The less straightforward but more robust approach: Download and install TeX

Installing (La)TeX on your personal computer means you don't need web access to edit your documents.
For Windows, you have a choice between MikTeX and TeX Live (I use and recommend MikTeX).
For Mac, there is MikTeX and MacTeX (I use and recommend MacTeX)
Every major Linux distribution bundles a TeX system (e.g., TeX Live) as an optional component—at the very least.

* MikTeX (Windows/Mac): [https://miktex.org/](https://miktex.org/)
* TeX Live (Windows/Linux): [https://www.tug.org/texlive/](https://www.tug.org/texlive/)
* MacTeX (Mac): [https://www.tug.org/mactex/](https://www.tug.org/mactex/)

## LaTeX tutorials and references

* [https://www.latex-tutorial.com](https://www.latex-tutorial.com)  
* Learn LaTeX in 30 minutes (tutorial at Overleaf):  
  [https://www.overleaf.com/learn/latex/Learn\_LaTeX\_in\_30\_minutes](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes)  
* Older and more detailed video tutorial at Overleaf (7 parts):  
  [https://www.overleaf.com/learn/latex/LaTeX\_video\_tutorial\_for\_beginners\_(video\_1)](https://www.overleaf.com/learn/latex/LaTeX_video_tutorial_for_beginners_\(video_1\))  
* List of LaTeX symbols:  
  [https://oeis.org/wiki/List\_of\_LaTeX\_mathematical\_symbols](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)
