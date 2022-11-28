# Final LHL project

Image SNR boosting / denoising using multiple noisy copies of the same image
## Goal
Given multiple noisy copies of the same image, explore ways to extract/reconstruct/estimate the true denoised image


## Dataset
[Urban 100 dataset](https://github.com/jbhuang0604/SelfExSR):
* https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip
* https://camo.githubusercontent.com/157add11addffa7acc9e9073e38ba386a03b906e1d9338f3944074d76e32dfcf/68747470733a2f2f756f66692e626f782e636f6d2f7368617265642f7374617469632f32306379396b6a69333939307079326a77753475776964686f337768326b65302e6a7067

Only the "_HR" High resolution images in the "image_SRF_2" directory was used.

## Approaches
* Center, average, then decenter
* Convolution
* PCA
* ICA
* Neural networks (autoencoders?)
    * REDNet
    * MWCNN
    * PRIDNet
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

## Loss metrics
OpenCV lists 5 different Image Quality Assessment (IQA) algorithms.
    * [BRISQUE](https://learnopencv.com/image-quality-assessment-brisque/): Blind/Referenceless Image Spatial Quality Evaluator
    * [GMSD](http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm): Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index
    * [MSE](https://en.wikipedia.org/wiki/Mean_squared_error): Mean Squared Error
    * [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio): Peak Signal to Noise Ratio
    * [SSIM](https://en.wikipedia.org/wiki/Structural_similarity): structural similarity index measure

GMSD and SSIM are metrics aimed at measuring the perceptual quality of images, while MSE and PSNR estimates absolute errors.
I will primarily use MSE and SSIM in evaluating results as those are the most popular metrics for image quality.
The built in QualityMSE and QualitySSIM classes from the opencv-python library will be utilised.

## Noise functions
skimage.util.noise methods will be used to generate the noisy images.

## Possible questions
* How do you measure noise in image? What metric?
* How does this compare with traditional denoising techniques where you don't have multiple copies?
## Associated terms
* [Brief review of image denoising techniques](https://vciba.springeropen.com/articles/10.1186/s42492-019-0016-7)
* [Denoising](https://www.iosrjournals.org/iosr-jece/papers/Vol.%2011%20Issue%201/Version-1/L011117884.pdf)
* [Noise in Digital Image Processing](https://medium.com/image-vision/noise-in-digital-image-processing-55357c9fab71)
* [Cross correlation](https://en.wikipedia.org/wiki/Cross-correlation)
* [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder/)
* [Denoising filtering techniques in findpeaks library](https://erdogant.github.io/findpeaks/pages/html/Denoise.html)
* [Image Denoising](https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
* [Smoothing Images](https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html)
* [Image denoising using deep learning](https://towardsai.net/p/deep-learning/image-de-noising-using-deep-learning)
* [Noise reduction](https://en.wikipedia.org/wiki/Noise_reduction#Removal)

## Tasks
- [x] Raw image dataset
- [x] Generate noise functions
- [x] Create noisy images
- [x] Choose SNR / loss metric
- [x] Implement loss metric function
- [x] Calculate baseline loss metric scores: ground truth vs noisy copies
- [x] Add PSNR metric
- [x] Test FastNL denoising with multiple copies
- [ ] Implement NNs
- [ ] Attempt LDA (useful for discrete data) on the assumption that pixels are discrete.
- [ ] Analyse distribution of lumenence and color data between true and noisy samples
- [ ] 
- [ ] 

## Project Structure
- [ ] Explain noise and properties
- [ ] Describe the data (distribution, and stats): truth vs noisy; lumenense vs color
- [ ] Approaches/Experiments
- [ ] Decomposition
- [ ] Filtering
- [ ] NN
- [ ] Comparison of results
- [ ] 


## Stretch goals:
- [ ] Dockerize
- [ ] Terraform deploy
- [ ] 


