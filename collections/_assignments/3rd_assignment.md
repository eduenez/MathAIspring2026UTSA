---
title: 3rd Assignment
ordinal: 3
due_date: 2026-02-18
---

## Warm-up Task
- (Re)read the [Written Work Guidelines]({{ site.baseurl }}{% link _refdocs/WrittenWorkGuidelines.md %}) and write a short paragraph reflecting on your progress as a technical writer.
Be specific; show that you understand your areas of strength and weakness better than you did at the beginning of the semester.

## Task I
Refer to the discussion of multivariate normal random variables (normally distributed random vectors) in §3.3, and also to the [Multivariate Normal Primer](https://peterroelants.github.io/posts/multivariate-normal-primer/) by [@peterroelants](https://peterroelants.github.io/).
You may and should use other resources in completing this task; high-quality reference textbooks are encouraged.

Consider an arbitrary n×n real matrix A.
The n×n matrix \\(S = A^{\top} A\\) is real symmetric and positive semi-definite.
(S is positive definite when its rank is exactly n.)
Therefore, S is a valid covariance matrix for an n-multivariate normal distribution;
in fact (apart from possible zero eigenvalues), the eigenvalues of S are the squares of the singular values of A, and the eigenvectors of S are the right-singular eigenvectors of A.

Write at least one full page explaining the precise relationship between the SVD of A and the ellipsoidal level curves of the normal PDF with covariance matrix S.
At the very least, your explanation should explain what are the exact directions and lengths of the (principal) axes of the ellipsoid with equation 
\\[
	\mathcal{N}(\mathbf{x}) 
	= 
	\exp\bigl(-Q(\mathbf{x})\bigr)/\sqrt{(2\pi)^n\det(S)}
\\]
where the positive semidefinite quadratic form \\(Q(\mathbf{x}) = (1/2)\mathbf{x}^\top S^{-1} \mathbf{x}\\).

### Bonus subtask A
What goes wrong if A —and therefore also S— is a singular matrix?

## Task II

### Subtask A
Let \\(\mathcal{N}\\) be the standard normal distribution \\(\tilde{\mathcal{N}} = \mathcal{N}\_{\mu,\sigma}\\) be the normal distribution with mean \\(\mu\\) and standard deviation \\(\sigma\\).
Evaluate the (Kullback-Leibler) KL-divergence \\(D_\mathrm{KL}(\mathcal{N}\parallel\tilde{\mathcal{N}})\\) in closed form.

*Hint:* Besides algebraic manipulations, all you should need are the values \\(\int_{-\infty}^{+\infty}\mathcal{N}(x)dx = 1\\), \\(\mathbb{E}(X) = \int_{-\infty}^{+\infty}x\mathcal{N}(x)dx = 0\\) and \\(\\mathbb{E}(X^2) = \int_{-\infty}^{+\infty}x^2\mathcal{N}(x)dx = 1\\).

For the remaining subtasks, let \\(B_p\\) be the PDF of a Bernoulli random variable with parameter \\(0\le p\le 1\\), i.e., \\(B_p(1) = p\\), \\(B_p(0) = 1-p\\) (and \\(B_p(x) = 0\\) for \\(x\ne 0, 1\\)).

Below, we assume \\(p,q\in[0,1]\\), i.e., the pair \\((p,q)\\) belongs to the unit square \\([0,1]\times[0,1] = [0,1]^2\\).

### Subtask B
Evaluate the KL divergence \\(D_{\mathrm{KL}}(B_p\parallel B_q)\\) in closed form (i.e., give a formula), and verify that \\(D_{\mathrm{KL}}(B_p\parallel B_{1-p}) = D_{\mathrm{KL}}(B_{1-p}\parallel B_p)\\).
Can you provide a conceptual explanation of the meaning of this equality?

### Optional Subtask C
Describe the regions of the unit square consisting of points \\((p,q)\\) where, respectively
- \\(D_{\mathrm{KL}}(B_p\parallel B_{q}) < D_{\mathrm{KL}}(B_{q}\parallel B_p)\\);
- \\(D_{\mathrm{KL}}(B_p\parallel B_{q}) = D_{\mathrm{KL}}(B_{q}\parallel B_p)\\);
- \\(D_{\mathrm{KL}}(B_p\parallel B_{q}) > D_{\mathrm{KL}}(B_{q}\parallel B_p)\\).
(Your description of those regions should be precise, but not necessarily be backed by formal proof —which will be messy.

An answer/conjecture found using numerical exploration/graphing is acceptable.)

<!-- (Recall that Colab allows opening files directly from [Github URLs](https://colab.research.google.com/github/){:target="_blank"}.) -->

<!-- Example: Use of colored labels. -->
<!-- Feb 30 -->
<!-- : **Lab**{: .label .label-purple } [Resizing Arrays](#) -->
<!-- : **HW 2 due**{: .label .label-red } -->
