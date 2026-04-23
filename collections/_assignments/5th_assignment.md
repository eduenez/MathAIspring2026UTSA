---
title: 5th Assignment
ordinal: 5
due_date: 2026-05-04
---

## Warm-up Task
- Look back at the writing reflection you submitted with your **1st Assignment**: 
The paragraph identifying a major error in your math writing and the steps you planned to take.
  - Quote or paraphrase what you wrote then. Has the improvement you described actually happened? Provide a brief "before" excerpt (from any of your earlier assignments this semester) and an "after" excerpt (from more recent work) as evidence.
  - Identify **one new issue** in your technical writing that you have only recently become aware of (i.e., something you did not notice at the start of the semester). This could be a strength you now want to push further, or a weakness that earlier problems were too simple to expose.
  - Keep this reflection to one well-crafted paragraph; precision and self-awareness matter more than length.

## Main Task: Choose Your Own Adventure

Your assignment is to solve some of the exercises from the interactive lessons on feedforward networks, backpropagation, deep network architectures, and generative models.
You do *not* need to solve every exercise: That is too much to ask!
Instead, this is a "choose your own adventure" assignment: Choose exercises that interest you and that push your understanding.

Format requirements are the same as for the 4th Assignment:
typeset in LaTeX at 12pt, single-spaced, 1 in. margins, using the provided [LaTeX Homework Template](https://github.com/eduenez/MathAIspring2026UTSA/blob/main/code/MathAI_HW_template.tex).

Your submission should be **at least 4 pages** (excluding the warm-up) and include **at least two figures**, drawn from at least two different lessons below.
You are allowed to use AI to generate figures as long as you include the exact prompt used; a vague prompt such as "make a figure about backpropagation" will not receive full credit.

**You must complete exercises from at least *two* among the subtasks I–IV below.**
<!-- (Subtask V is optional/extra credit but recommended.) -->

## Subtask I: Units, Activations, and the Forward Pass

Open the [Forward Pass & Activation Functions lesson](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/fwdpass_units.ipynb) in Colab and follow it through.
Choose and solve some of the exercises listed in Sections §2 and §4:

- **Exercise 2.1** *(Linear networks collapse)*: Extend the two-layer linearity check to $L = 5$ layers; analyze a non-identity linear activation; state what the Universal Approximation Theorem says.
- **Exercise 4.1** *(Counting parameters)*: Derive the general parameter-count formula; apply it to the MNIST architecture; compare widths 64 vs. 128.
- **Exercise 4.2** *(Choosing activations)*: Justify the correct output activation for single-label classification, multi-label detection, and regression.
- **Exercise 4.3** *(Vanishing gradients: Experiment)*: Run the gradient-norm scaffold with sigmoid and ReLU; explain the difference; find an initialization scale that stabilizes sigmoid.

## Subtask II: Backpropagation

Open the [Backpropagation lesson](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/backpropagation.ipynb) in Colab.
Choose and solve some of the exercises in Sections §3–§5:

- **Exercise 3.1** *(Chain rule on the graph)*: Verify \\(\partial\ell/\partial w\\), \\(\partial\ell/\partial b\\), \\(\partial\ell/\partial x\\) by hand; redo the backward pass for a sigmoid unit; explain why large-magnitude inputs lead to large weight updates.
- **Exercise 4.1** *(Deriving \\(\delta^{(L)}\\))*: Derive the output-error formula \\(\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y}\\) from the softmax and cross-entropy expressions.
- **Exercise 4.2** *(Hidden-layer recurrence)*: Fill in the full derivation of \\(\nabla_{W^{(l)}}\mathcal{L} = \delta^{(l)}(\mathbf{a}^{(l-1)})^\top\\); interpret the ReLU backward pass in terms of dead units.
- **Exercise 5.1** *(Extending to a batch)*: Generalize the forward/backward code to a batch of \\(B\\) examples; verify that one gradient step decreases the loss.
- **Exercise 5.2** *(Sigmoid network)*: Swap ReLU for sigmoid, re-run the gradient check, and compare \\(\delta^{(1)}\\) magnitudes between the two activations.

## Subtask III: Deep Network Architectures

Open the [DNN Architectures Overview](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/dnn_architectures_overview.ipynb) in Colab.
Follow the lesson through its survey of MLP, CNN, RNN, Transformer, and autoencoder families, and complete exercises of your choice.
As a guide, here are the kinds of questions worth addressing:

- Compare the parameter count and inductive biases of a fully-connected layer vs. a convolutional layer of similar "capacity."
- Explain, in your own words, why a recurrent network can in principle process sequences of arbitrary length, while a fixed-window feedforward network cannot.
- Describe the self-attention mechanism: What are queries, keys, and values, and how does the scaled dot-product formula produce an attention matrix?
- Explain the role of the encoder and decoder in an autoencoder; contrast a plain autoencoder's latent space with that of a VAE (connecting to Subtask IV below).

## Subtask IV: Generative Models

Open the [Generative Models lesson](https://colab.research.google.com/github/eduenez/MathAIspring2026UTSA/blob/main/code/generative_models.ipynb) in Colab.
The exercises are listed at the end of the notebook (Section §7); choose and solve some of them:

1. **ELBO derivation**: Use Jensen's inequality to derive the ELBO; 
show that the gap equals 
\\(
D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x})\|p_\theta(\mathbf{z}|\mathbf{x}))
\\).
2. **KL weight experiment**: Train the VAE with `kl_weight` values 0.01, 0.1, 1.0, 10.0; visualize and explain the reconstruction vs. regularization tradeoff.
3. **Temperature and entropy**: Prove that the entropy of the temperature-scaled softmax is monotone in \\(\tau\\); verify empirically.
4. **Autoregressive generation speed**: Compare generation cost (forward passes) between autoregressive models and VAEs; discuss quality tradeoffs.
5. **PCA vs. VAE on the circle**: Explain why a 1-D VAE latent space is topologically insufficient for the circle; determine the minimum latent dimension for a torus (surface of a doughnut).

<!-- ## [*Extra Credit*] Optional Subtask V: Synthesis -->

<!-- Choose **one** of the following synthesis problems connecting ideas across the lessons above: -->

<!-- **(A)** The gradient of the cross-entropy loss with respect to the output logits is \\(\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y}\\) (backpropagation lesson). Show that this is also the gradient of the *negative ELBO reconstruction term* in a VAE with a categorical decoder. What does this tell you about the relationship between supervised classification and generative modeling? -->

<!-- **(B)** A plain autoencoder (from the architectures lesson) and a VAE (from the generative models lesson) both consist of an encoder and a decoder. Identify the precise mathematical differences between the two objectives, and explain: In terms of the geometry of the latent space: Why only the VAE can be used to generate new samples by drawing \\(\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)\\). -->

<!-- Example: Use of colored labels. -->
<!-- Feb 30 -->
<!-- : **Lab**{: .label .label-purple } [Resizing Arrays](#) -->
<!-- : **HW 2 due**{: .label .label-red } -->
