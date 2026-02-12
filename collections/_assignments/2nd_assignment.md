---
title: 2nd Assignment
ordinal: 2
due_date: 2026-02-18
---

## Warm-up Task
- (Re)read the [Written Work Guidelines]({{ site.baseurl }}{% link _refdocs/WrittenWorkGuidelines.md %}) and write a short paragraph reflecting on your progress as a technical writer.

## Main Task I

Log on to [Colab](https://colab.research.google.com/){:target="_blank"} and open the [Jupyter (iPython) file `numpy-tutorial-svd.ipynb`](https://colab.research.google.com/drive/1A5U2jgdtyVQZKHIIpqHub6X-aUqD-7mB?usp=sharing).

Make sure to save it to your own Google Drive!
(The link above is to a read-only version of my personal copy of the file).
Afterwards, you should continue work on your own personal copy.

Take a “random” 3×2 matrix A of your own choosing meeting the following requirements:
- it has small integer entries, say 0, ±1, ±2, ±3
  (repetitions are allowed, and you are also allowed to use larger integers but the calculations may get messier);
- it has rank 2;
- no two rows, nor columns should be perpendicular;
- at most two entries are zero, but not in the same row nor column.

### Subtask 1 [50 pt.]
Run through the entire calculation of finding the Singular Value Decomposition of A.

Your calculations should all be *exact*—not decimal approximations.
Square roots are almost certain to be involved in computing eigenvalues and entries of eigenvectors.
Do not evaluate those numerically, but *do* simplify the exact expressions as much as possible
(e.g., although square roots of relatively large integers may appear, at *no* point should you have a square root nested inside another one, etc.)

The steps are as follows:
- Consider first the 2×2 matrix \\(B = A^{\top}A\\).
	Carry out in full all steps for finding the (positive!) eigenvalues \\(s_1\\), \\(s_2\\) (chosen in the *decreasing* order of magnitude \\(s_1 > s_2\\)) and an orthonormal basis \\(v_1\\), \\(v_2\\) (ONB) of eigenvectors of B —these are the “right eigenvectors” of A.

The eigenvalues  \\(s_1 > s_2\\) of B are the *singular values* of A.
  
  The ONB \\(v_1\\), \\(v_2\\) gives (the columns of) an orthogonal 2×2 matrix \\(V\\).
  Verify that \\(V^\top V = I\\).
- Consider now the 3×3 matrix \\(C = AA^{\top}\\).
  Carry out all steps of finding the eigenvalues and an orthonormal basis \\(u_1\\), \\(u_2\\), \\(u_3\\) (ONB) of eigenvectors for C—these are the “left eigenvectors” of A.
  Verify that the corresponding eigenvalues should be \\(s_1\\), \\(s_2\\) (which are the *same* singular values of A as above) and the third eigenvalue is \\(0\\).
  The ONB \\(u_1\\), \\(u_2\\), \\(u_3\\) gives (the columns of) an orthogonal matrix \\(U\\).
  Verify that again \\(U^\top U = I\\).
- Let S be the 3×2 matrix (same size as A) having the singular values \\(s_1\\), \\(s_2\\) on the diagonal.
 Verify that
 \\[
 s_1(u_1v_1^\top) + s_2(u_2v_2^\top) = A = USV^\top.
 \\]
 
### Subtask 2 [25 pt.]
1. Find numerical approximations (using a calculator or Python/NumPy) to the matrices U, S, V you found above, and write  them down.
2. Use NumPy to find (a decimal approximation to) the SVD of the same matrix A chosen above.
   (*Hint:* Use NumPy's function `linalg.svd` as done in [`numpy-tutorial-svd.ipynb`](https://colab.research.google.com/drive/1A5U2jgdtyVQZKHIIpqHub6X-aUqD-7mB?usp=sharing).
   Remember to start with `import numpy as np` so you can subsequently access the function as `np.linalg.svd`.)
3. Do the matrices U, S, V you found in (i) above answer agree exactly (to within a small rounding error) with the ones found using `linalg.svd`?
   How many different choices of the matrix triple (U, S, V) are possible, all of which are equally correct as an SVD for A?
4. [Bonus +5%] Count the possible choices of (U, S, V) for a given matrix A still of rank 2, but having size 4×2.
   
### Subtask 3 [25 pt.]
1. As in Cell #25 of [`numpy-tutorial-svd.ipynb`](https://colab.research.google.com/drive/1A5U2jgdtyVQZKHIIpqHub6X-aUqD-7mB?usp=sharing), 
   let k be any positive integer not exceeding 768, which is the rank, in all likelihood exact, of the “raccoon matrix” `img_gray`.
   In `numpy-tutorial-svd.ipynb`, only the largest k = 10 singular values (out of 768) are kept, providing a “lossy”/blurry reconstruction of the original image which, remarkably, is still quite recognizable as a raccoon.
   Carry out experiments varying the value of k and report your findings.
   What is the smallest value of k for which there is no visually discernible difference between the original picture and the reconstructed one?
   
Although computing the SVD of a very large matrix is quite an expensive operation for which no truly efficient algorithm is known, the notions of “encoder/decoder” in modern machine learning (deep neural networks) are inspired by (but not based on) optimal/ideal mathematical procedures such as SVD.

### Bonus subtask 4 [15 pt.]
Read about the relation between SVD and PCA (Principal Component Analysis) in section §2.12 of the textbook and (at least one) additional source(s).
Write a summary —with illustration(s)!— of your findings and spanning about a half-page.

<!-- (Recall that Colab allows opening files directly from [Github URLs](https://colab.research.google.com/github/){:target="_blank"}.) -->

<!-- Example: Use of colored labels. -->
<!-- Feb 30 -->
<!-- : **Lab**{: .label .label-purple } [Resizing Arrays](#) -->
<!-- : **HW 2 due**{: .label .label-red } -->
