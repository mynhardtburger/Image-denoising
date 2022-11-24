# Final LHL project

Image SNR boosting / denoising using multiple noisy copies of the same image
## Goal
Given multiple noisy copies of the same image, explore ways to extract/reconstruct/estimate the true denoised image


## Dataset
[Smartphone Image Denoising Dataset](https://www.eecs.yorku.ca/~kamel/sidd/index.php)
Smaller resized versions, with fixed dimensions, will be used for the initial work to spead up processing.

## Approaches
* Center, average, then decenter
* Convolution
* PCA
* ICA
* Neural networks (autoencoders?)
* Other decomposition techniques?

## Tools
* Image Noise generator: https://en.wikipedia.org/wiki/Image_noise
    * AWGN: Additive white gausian noise
    * Salt & pepper noise
    * Multiplicative noise
    * https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    * https://scikit-image.org/docs/stable/api/skimage.util.html#random-noise
    * https://theailearner.com/2019/05/07/add-different-noise-to-an-image/  
* Image sample generator

## Possible questions
* How do you measure noise in image? What metric?
## Associated terms
* [Brief review of image denoising techniques](https://vciba.springeropen.com/articles/10.1186/s42492-019-0016-7)
* [Denoising](https://www.iosrjournals.org/iosr-jece/papers/Vol.%2011%20Issue%201/Version-1/L011117884.pdf)
* [Noise in Digital Image Processing](https://medium.com/image-vision/noise-in-digital-image-processing-55357c9fab71)
* [Cross correlation](https://en.wikipedia.org/wiki/Cross-correlation)
* [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder/)

## Tasks
- [ ] Raw image dataset
- [ ] Resize image dataset
- [ ] Generate noise functions
- [ ] Create noisy images
- [ ] Choose SNR / loss metric
- [ ] Implement loss metric function
- [ ] Calculate baseline loss metric scores: ground truth vs noisy copies
- [ ] blank
- [ ] blank