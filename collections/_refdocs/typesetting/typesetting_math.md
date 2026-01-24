---
layout: page
title: Typesetting Mathematics
nav_exclude: true
output: true
description: >-
    Guide to typesetting mathematics
---

# Guide to typesetting mathematics

A curated list of resources and recommendations.

Common text processors like MS Word, Pages, and OpenOffice Writer are general-purpose tools poorly suited for writing documents containing many mathematical formulas.  There are much better, specialized professional tools available to write math.  This Guide seeks to help you choose the tool that best suits your needs.

# To TeX, or not to TeX, that is the Question

The 800-pound gorilla in the room of would-be math authors is the TeX system (pronounced: *“Tek”*, with apologies to Texans, never *“Teks”*\!).  *TeX is to math writing as Word is to word processing:*  It is the default program with the largest market share, considered the standard in its class, and to which everything else is compared.  TeX (or LaTeX, pronounced *“Lah-Tek”*) is a system that will likely accompany you as you progress through your study of mathematics and other sciences, but it is not the only high-quality typesetting system for math.  The advantages and disadvantages of TeX vs. non-TeX typesetting are discussed in the sections [To TeX](#option-1:-to-tex!) and [Not To TeX](#option-2:-not-to-tex!) below.

## Option 1: To TeX\! {#option-1:-to-tex!}

Writing TeX is a bit like writing a simple computer program.  For instance, to display the formula   
c \= a2+b2,  
you actually need to type  
c \= \\sqrt{a^2 \+ b^2}  
in a regular text file (a plain text file—the type you create using Notepad in Windows or TextEdit in Mac OS, rather than a rich-text file such as a Word .DOC or DOCX document).  After composing the plan text file and giving it the extension .TEX, a second step is necessary in order to create a PDF file that actually displays the formula.  This probably sounds very strange if you have never written computer code followed by the additional step of compiling the code to obtain the final product (an executable program).  The TeX system (and the extremely popular extension of TeX known as **LaTeX[^1]**) is the professional typesetting tool used by more mathematicians and companies publishing technical manuscripts with a lot of math (also, by lots of computer scientists, and even musicians who need to typeset sheet music).

In practice, TeX and LaTeX have become a lot easier to use, and there are also excellent tutorials available.  Moreover, if you want the level of quality achievable with LaTeX but don’t really want to code TeX, there is an excellent desktop app called LyX that allows writing LaTeX documents with no unfamiliar coding at all.  LyX is a bit of a lookalike to Word or Pages but focused on math.  One disadvantage of LyX is that installing it is a two-step process: First, you need to install a TeX system in your computer (such as MikTeX, MacTeX, or TeXLive), and only afterwards is LyX itself installed.

If you want to learn LaTeX and use it directly, you’ll find it’s actually pretty easy once you get used to the basics.  The easiest way to write LaTeX nowadays is on the web (once upon a time, the only way to use TeX/LaTeX was by installing it on one’s computer).  You can use a minimal, lightway editor like StackEdit[^2], which is more than adequate for relatively short documents (like a homework assignment or exam, but not for, say, a final paper), or a fully fledged LaTeX online system like Overleaf, which is free for personal, non-collaborative usage (that’s all you need for this class).

## Option 2: Not to TeX\! {#option-2:-not-to-tex!}

Okay, discussion of the 800-pound gorilla (TeX/LaTeX) is over.  Alternatives to Word are much less well-known, but can still be excellent for certain purposes.  The same is true of alternatives to TeX.  I will give only one recommendation that is, in my opinion, a standout.  It is not nearly as well known as any TeX/LaTeX system, but the editor **TeXmacs** is an outstanding typesetting system in itself. (TeXmacs is pronounced *“Tek-macs”*, as you would expect from the pronunciation of TeX as *“Tek”.*)  If you are willing to spend an hour or two following TeXmacs tutorials, you may find that it represents a great balance of two simultaneous outcomes: 

* (*i*) high-quality typesetting of both text and math, and   
* (*ii*) efficiency and ease of creation of documents themselves.

It is really hard to balance the two goals above.  For instance, TeX/LaTeX are superb at (*i*) but mostly fail at (*ii*).  (A LaTeX beginner will have a hard time producing truly well-formatted and nice-looking documents; this is because of LaTeX’s weakness in (*ii*).)  Word excels at neither:  It does not produce high-quality typesetting (math or otherwise) and, while creating documents is easy if they are only text, Word is very clunky when it comes to creating math-rich content, despite the fact that it has a competent formula editor.  A LaTeX-aware markdown editor like StackEdit scores high on (*ii*), but not so much on (*i*), and is otherwise only suited for short documents.

One key feature of TeXmacs is the choice of ways of entering mathematical formulas and symbols given to the user.  There are usually three or four different ways of entering a symbol.    
For instance, when TeXmacs is in math-input mode, the Greek letter alpha (α) may be entered into the document in four different ways:

1. Clicking on the “Insert a Greek character” toolbar icon (that displays the symbol Γ, the capital Greek letter gamma), then the icon for the letter α on the pop-up menu displayed.  
2. From the menu system: Insert/Symbol/Greek letter/α.  
3. Typing “a” followed by the Tab key.  
4. (LaTeX-style) Typing \\alpha followed by Enter.

TeXmacs tutorials will almost always use the shortcut “a˽Tab” to type α, because it is most efficient, but there is no reason (especially when first learning) to use either of the other ways.

# Finally: Recommendations\!

## For the tech-savvy, or those with programming experience:  Full-speed ahead LaTeX

If this is you, the choice is clear: Go ahead and follow some LaTeX tutorials.  Practice online at Overleaf, and progress to writing your documents there.  If you want to install TeX/LaTeX on your own computer so you can write offline using a LaTeX text editor such as TeXMaker, WinEdt, TeXworks, TeXstudio or TeXShop (instead of Overleaf on the cloud) install MikTeX, MacTeX, or TeXLive and the editor of your choice.

## For those who want to ease into LaTeX

* If you just want to slowly learn LaTeX commands (such as \\sqrt for a square root, \\alpha for the Greek letter α) in the least-overhead environment possible, use StackEdit.  Read the Markdown and KaTeX tutorials (KaTeX is a subset of LaTeX that is supported on web apps like StackEdit).  
* If you want a Word-like experience based on LaTeX, use LyX.  The installation is a two-step process that will not work if you are an “I want it all set-up with one-click” computer user.  You will still need to follow online tutorials and read documentation.  Installing and using LyX is still one notch more cumbersome than TeXmacs below, but you are easing into LaTeX.  You will also get much better-looking documents, without excessive tech wizardry, than if you use LaTeX by itself (at Overleaf or otherwise).  \[However, the champ of good-looking easy-to-create documents is TeXmacs—read below.\]

## For rebels going against the tide who still want it all

Give TeXmacs a try.  All you need is to download the package, do a single-click install, and follow tutorials for an hour or two.  No need to install anything separately because TeXmacs is its own complete ecosystem independent of TeX\!  Okay, so TeXmacs is not TeX/LaTeX. It is not well-known in comparison. But, yes, it will give you the best-looking and most easily composed documents around.  (If you show TeXmacs documents to LaTeX users, they’ll often ask: What LaTeX class did you use and how did you get fonts to look that pretty?  Answer: It’s not LaTeX, it’s TeXmacs\!)

## Afterword: How does the instructor write math?

I started using TeX (more specifically, AMSTeX, back when it was still more prevalent than LaTeX) as an undergrad under the tutelage of one of my professors.  My undergraduate thesis was written in AMSTeX. As a grad student, tools like LyX and TeXmacs were recent and less polished, plus I had already learned AMSTeX, and easily migrated to LaTeX.  Since writing my PhD thesis, I have been using LaTeX almost exclusively, using emacs as editor, which I still do nowadays.

# Resources

* Overleaf (online LaTeX editor): [https://www.overleaf.com](https://www.overleaf.com/)  
* StackEdit (online Markdown editor with KaTeX support): [https://stackedit.io/](https://stackedit.io/)  
* LyX home page: [https://www.lyx.org/](https://www.lyx.org/)  
* TeXmacs home page: [https://www.texmacs.org/](https://www.texmacs.org/)

## Full TeX/LaTeX systems (mostly not needed: read first\!)

These are only needed if you want to install TeX directly on your computer (for offline use).  A TeX system is needed as a first-step before installing LyX (for instance), but *not needed* if you are using Overleaf or any of the other resources above.  For Windows, you have a choice between MikTeX and TeX Live.  For Mac, there is MikTeX and MacTeX (which is basically TeX Live for Mac). 

* MikTeX (Windows/Mac): [https://miktex.org/](https://miktex.org/)  
* TeX Live for Windows: [https://www.tug.org/texlive/](https://www.tug.org/texlive/)  
* MacTeX for Mac: [https://www.tug.org/mactex/](https://www.tug.org/mactex/)

 I use MikTeX in Windows computers, MacTeX in Macs.

## LaTeX tutorials and references

(Both LyX and TeXmacs come with excellent built-in documentation, which is yet another argument for choosing them over plain LaTeX .)

* [https://www.latex-tutorial.com](https://www.latex-tutorial.com)  
* Learn LaTeX in 30 minutes (tutorial at Overleaf):  
  [https://www.overleaf.com/learn/latex/Learn\_LaTeX\_in\_30\_minutes](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes)  
* Older and more detailed video tutorial at Overleaf (7 parts):  
  [https://www.overleaf.com/learn/latex/LaTeX\_video\_tutorial\_for\_beginners\_(video\_1)](https://www.overleaf.com/learn/latex/LaTeX_video_tutorial_for_beginners_\(video_1\))  
* List of LaTeX symbols:  
  [https://oeis.org/wiki/List\_of\_LaTeX\_mathematical\_symbols](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)

## Miscellaneous links

* List of KaTeX symbols. (Also applies to LaTeX and to StackEdit documents.):  
  [https://katex.org/docs/supported.html](https://katex.org/docs/supported.html)  
* Markdown cheat sheet (useful for StackEdit users):  
  [https://www.markdownguide.org/cheat-sheet/](https://www.markdownguide.org/cheat-sheet/)  
* Guide to using LyX for homework: [https://pj.freefaculty.org/guides/Computing-HOWTO/LatexAndLyx/LyX-for\_LaTeX\_homework/LyX-LaTeX\_homework.pdf](https://pj.freefaculty.org/guides/Computing-HOWTO/LatexAndLyx/LyX-for_LaTeX_homework/LyX-LaTeX_homework.pdf)  
* LyX wiki (guides, tips, howtos, etc.)  
  [https://wiki.lyx.org/](https://wiki.lyx.org/)

## FAQ and How-To

### How do I convert a StackEdit document to PDF?

You should export your document as HTML (Styled HTML), save it someplace it, then open it on the web browser (for instance, by double-clicking the HTML file), and Print the file from the browser, BUT from the Print dialog box (in Windows) choose “Save as PDF” or “Print to PDF” instead of choosing a physical printer.  (In Mac, use the PDF button on the Print dialog box and choose “Export to PDF”).  You will be asked for a filename and location where to save a PDF file.

Alternatively, you can become a paid sponsor of StackEdit, which enables its “Export as PDF” feature, but costs $5 for 3 months, or $15/year.

### TeXmacs seems to take a long time to respond or seems to hang

The first few times you use TeXmacs, when changing a document’s default font, and possibly at the very beginning of editing a document, lots of things are happening behind the scenes. Be patient. I have a fast desktop computer, yet the first few times I started editing in TeXmacs it would take several seconds to perform certain operations (such as inserting a math formula for the first time). Be patient and let TeXmacs do its initialization jobs. Subsequently editing the rest of your document should be pretty snappy with only the occasional delay.

### Help\!  I’m really stuck\!

The instructor does not have the means to provide technical support.  I may be able to give hints or pointers, but almost everything I have to offer in the way of insight is contained in this document. Please read it carefully before giving up in despair  (Also, a tea or coffee break may help. Computers are just temperamental sometimes.)

[^1]:  Actually, it has become common to use TeX as a synonym for LaTeX, but this is really doing a disservice to the TeX system that serves as basis for many others, not only LaTeX, but also ConTeXt, AMSTeX, and extensions of the system itself such as pdfTeX, XeTeX, LuaTeX, …. The list of different “TeX’es” is long.

[^2]:  Note that StackEdit is not strictly speaking a LaTeX editor per se.  Instead, it’s a Markdown editor that accepts and renders LaTeX formulas.  Markdown is perhaps the simplest way to generate rich-text documents on the web, and StackEdit is a superb implementation of Markdown+LaTeX.
