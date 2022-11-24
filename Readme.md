# Final LHL project

Denoising / artifact removal

## Associated terms
* https://vciba.springeropen.com/articles/10.1186/s42492-019-0016-7
* [Denoising](https://www.iosrjournals.org/iosr-jece/papers/Vol.%2011%20Issue%201/Version-1/L011117884.pdf)
https://medium.com/image-vision/noise-in-digital-image-processing-55357c9fab71


## Dataset
Source of images

## Goal
Given multiple samples of mixed audio for the same underlying event, are we able to decompose and extract the original source audio samples?

## Possible questions
* How to measure accuracy/similarity of extracted signal vs source signal
* What combinations present complications. What type of elements mostly impact accuracy?
* Can signal preprocessing be used to generated additional mix samples allowing either:
    1. more accurate extraction
    2. extraction of more distinct signals
* Can you encode and extract data within signals? Given the same message encoded in multiple samples, can the common signal be extracted?

## Methods
* FASTICA
* PCA
* [JADE](https://en.wikipedia.org/wiki/Joint_Approximation_Diagonalization_of_Eigen-matrices)

## Links
* https://mscipio.github.io/post/bss-shogun-python/
* https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e
* https://towardsdatascience.com/independent-component-analysis-ica-a3eba0ccec35
* http://www.mit.edu/~gari/teaching/6.555/LECTURE_NOTES/ch15_bss.pdf
* https://medium.com/pytorch/addressing-the-cocktail-party-problem-using-pytorch-305fb74560ea
* https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html

