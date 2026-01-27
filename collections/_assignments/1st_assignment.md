---
title: 1st Assignment
ordinal: 1
due_date: 2026-02-04
---

### Warm-up Task
- Read the [Written Work Guidelines]({{ site.baseurl }}{% link _refdocs/WrittenWorkGuidelines.md %}) and documents on mathematical writings referenced therein.
  Write a short paragraph (≈5 lines) reflecting on one major error present in your math writing hitherto, and outlining some steps you will take to improve your writing for this course.
  Each written assignment throughout the semester should start with a short 1-paragraph reflection on a *different* issue you have noticed, and outline steps to address it.
  (You will need to continuously revisit the Written Work Guidelines and documents therein referenced.)

### Task I

Log on to [Colab](https://colab.research.google.com/) and open the Jupyter (iPython) file `mnist_ff.ipynb`   which creates and trains a simple feedforward-network to learn to recognize handwritten digits in the MNIST “database”—which is more of a *dataset* than a *database* as such.
(The file is hosted at Github [https://eduenez.github.io/MathAIspring2026UTSA/code/mnist_ff.ipynb).](https://eduenez.github.io/MathAIspring2026UTSA/code/mnist_ff.ipynb) 
It is not even necessary to download it! Colab allows opening files directly from Github URLs.)\\
Make sure to look at the [MNIST Wikipedia page](https://en.wikipedia.org/wiki/MNIST_database) and Figure 1.9 in the [*Deep Learning.*](http://www.deeplearningbook.org){:target="_blank"} book.
- Make sure to *Save your own personal copy* of the file (e.g., to your Google Drive).
  - You *cannot* save it back to Github (not authorized).
- Although the Python code is somewhat complicated, it has many comments.
  There are also a couple of parameters whose numerical value you can adjust interactively 
  (but you must re-execute the notebook for those changes to take effect).

#### Sub-Task 1
Play around assigning different values to the parameters `dim_hl_1`, `dim_hl_2` (dimensionalities of the 1st and 2nd hidden layers) and the number `num_epochs` of epochs.
Make a table and provide a human-readable analysis of your findings.

For extra credit +20%, also compare the 2-hidden layer feedforward NN to NNs with 1 or 3 hidden layers.

#### Sub-Task 2
There are probably at least a few new and unfamiliar concepts present in or implied by the code.
Choose one aspect or issue you would like to understand better, research it, and write about 1 page summarizing your findings.
Examples include (but are not limited to):
- What is a TPU? What are its similarities and differences to a CPU and a GPU?
- What is Keras? What is Jax? What is NumPy? How are they related?
- What is a `float32` precisely? How are those 32 bits used to store real numbers?
- What other kinds of floats and numerical types does NumPy support?
- What does a Keras model actually consist of? What does it look like as a Python object?
- What is a "logit"? What is a “ReLU"? What is "categorical cross-entropy"?
- What are other datasets beyond MNIST broadly used as NN benchmarks?
- What is the difference between training, validation and test datasets?
- What is the difference between a feedforward network and other kinds of neural networks?
- What is root mean-square propagation? How does it compare to other optimization methods?
- What part of the code in `mnist_ff.ipynb` do I find most mysterious?
  Can I try to better understand and explain what it's doing?


<!-- Example: Use of colored labels. -->
<!-- Feb 30 -->
<!-- : **Lab**{: .label .label-purple } [Resizing Arrays](#) -->
<!-- : **HW 2 due**{: .label .label-red } -->
