# audio_source_separation
An implementation of audio source separation tools.

## Nonnegative Matrix Factorization
| Method | References | Example |
|:-:|:-:|:-:|
| NMF | ["Algorithms for Non-Negative Matrix Factorization," D. D. Lee et al., 2000](https://dl.acm.org/doi/10.5555/3008751.3008829) <br> ["Nonnegative Matrix Factorization with the Itakura-Saito Divergence: With Application to Music Analysis," C. Févotte et al., 2009](https://ieeexplore.ieee.org/document/6797100) | EUC-NMF: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/nmf-example/euc-nmf/test_euc-nmf.ipynb) <br> KL-NMF: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/nmf-example/kl-nmf/test_kl-nmf.ipynb) <br> IS-NMF: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/nmf-example/is-nmf/test_is-nmf.ipynb) <br> t-NMF: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/nmf-example/t-nmf/test_t-nmf.ipynb) <br> Cauchy-NMF: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/nmf-example/cauchy-nmf/test_cauchy-nmf.ipynb) |
| Complex NMF | ["Complex NMF: A New Sparse Representation for Acoustic Signals," H. Kameoka et al., 2009](https://ieeexplore.ieee.org/document/4960364) | EUC-CNMF: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/nmf-example/cnmf/test_euc-cnmf.ipynb) |

## Blind Source Separation
| Method | References | Example |
|:-:|:-:|:-:|
| Multichannel NMF (MNMF) | ["Multichannel Nonnegative Matrix Factorization in Convolutive Mixtures for Audio Source Separation," A. Ozerov and C. Fevotte, 2009](https://ieeexplore.ieee.org/document/5229304) <br> ["Multichannel Extensions of Non-Negative Matrix Factorization With Complex-Valued Data," H. Sawada et al., 2013](https://ieeexplore.ieee.org/document/6410389) | IS-MNMF: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/mnmf/test_is-mnmf.ipynb) |
| FastMNMF | ["Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices," K. Sekiguchi et al., 2019](https://arxiv.org/abs/1903.03237) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/mnmf/test_fast-mnmf.ipynb) |
| FDICA | ["An Approach to Blind Source Separation Based on Temporal Structure of Speech Signals," N. Murata et al., 2001](https://www.sciencedirect.com/science/article/abs/pii/S0925231200003453) <br> ["Underdetermined Convolutive Blind Source Separation via Frequency Bin-Wise Clustering and Permutation Alignment," H. Sawada et al., 2011](https://ieeexplore.ieee.org/document/5473129) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/fdica/test_fdica.ipynb) |
| IVA (Gradient descent) | ["Independent Vector Analysis: An Extension of ICA to Multivariate Components," T. Kim et al., 2006](https://link.springer.com/chapter/10.1007/11679363_21) <br> ["Solution of Permutation Problem in Frequency Domain ICA, Using Multivariate Probability Density Functions," A. Hiroe, 2006](https://link.springer.com/chapter/10.1007/11679363_75) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/iva/test_grad-iva.ipynb) |
| AuxIVA | ["Stable and Fast Update Rules for Independent Vector Analysis Based on Auxiliary Function Technique," N. Ono, 2011](https://ieeexplore.ieee.org/document/6082320) <br> ["Auxiliary-function-based Independent Vector Analysis with Power of Vector-norm Type Weighting Functions," N. Ono, 2012](https://ieeexplore.ieee.org/document/6411886) <br> ["Fast and Stable Blind Source Separation with Rank-1 Updates," R. Scheibler and N. Ono, 2020](https://ieeexplore.ieee.org/document/9053556) | AuxIVA-IP: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/iva/test_aux-iva-ip.ipynb) <br> AuxIVA-ISS: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/iva/test_aux-iva-iss.ipynb) |
| ILRMA | ["Determined Blind Source Separation Unifying Independent Vector Analysis and Nonnegative Matrix Factorization," D. Kitamura et al., 2016](https://ieeexplore.ieee.org/document/7486081) <br> ["Faster Independent Low-Rank Matrix Analysis with Pairwise Updates of Demixing Vectors," T. Nakashima et al., 2021](https://ieeexplore.ieee.org/document/9287508) | Gauss-ILRMA: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/ilrma/test_gauss-ilrma.ipynb) <br> t-ILRMA: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/ilrma/test_t-ilrma.ipynb) <br> Gauss-ILRMA-IP2: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/ilrma/test_gauss-ilrma-ip2.ipynb)|
| Consistent-ILRMA | ["Consistent Independent Low-rank Matrix Analysis for Determined Blind Source Separation," D. Kitamura and K.Yatabe, 2020](https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-020-00704-4)| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/audio_source_separation/blob/main/egs/bss-example/ilrma/test_consistent-ilrma.ipynb) |