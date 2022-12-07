# Image denoising
I wanted to explore image noise and some interesting denoising methods. This was done as a final project for my Lighthouse labs datascience diploma.
What I found that the available tools and methods is vast and this in its own right is a speciality subfield that spans many industries. I ended up exploring only a small subset of approaches, mostly drawn to those which were easy to implement and understand for a beginner like me.

The lessons I learned from this high level investigation are:
1. There are many different types of noise with different statistical characteristics. As a result there is no single best approach.
2. Measuring the level of noise in an image can be difficult. Different metrics prioritize different things and sometimes you don't have a true/clean image to compare your results against.
3. Computational complexity of the approaches spans the whole gamut, from fast image smoothing methods to slow and bulky neural networks.

I wish I had more time to explore techniques which can extract a clean image from multiple sources/copies (eg. image fusion). These seem to be extremely powerful but often limited to niche areas such as astrophotography.
Why is that?  Could this approach be adapted to be more flexible?

## Notebooks & results
1. Impact of noise on color distribution profiles and generating baseline noise data: [link](./notebooks/EDA.ipynb)
2. Testing image smoothing, non-local means and image fusion methods: [link](./notebooks/Traditional%20techniques.ipynb)
3. Implementation of Deep Image Prior CNN: [link](./notebooks/Deep%20Image%20Prior.ipynb)
4. Graphing of results: [link](./notebooks/Results.ipynb)
5. Presentation slides: [link](./Presentation.pptx)

## Dataset
Urban 100 dataset:
* https://github.com/jbhuang0604/SelfExSR
* https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip
* https://camo.githubusercontent.com/157add11addffa7acc9e9073e38ba386a03b906e1d9338f3944074d76e32dfcf/68747470733a2f2f756f66692e626f782e636f6d2f7368617265642f7374617469632f32306379396b6a69333939307079326a77753475776964686f337768326b65302e6a7067

Only the "_HR" High resolution images in the "image_SRF_2" directory was used.

## Useful and interesting resources
* [Wikipedia - Noise reduction](https://en.wikipedia.org/wiki/Noise_reduction#Removal)
* [Wikipedia - Image fusion](https://en.wikipedia.org/wiki/Image_fusion)
* [Brief review of image denoising techniques](https://vciba.springeropen.com/articles/10.1186/s42492-019-0016-7)
* [Image Denoising Techniques-An Overview](https://www.iosrjournals.org/iosr-jece/papers/Vol.%2011%20Issue%201/Version-1/L011117884.pdf)
* [OpenCV tutorial - image smoothing](https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html)
* [OpenCV tutorial - image denoising](https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html)
* [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior)
* [TF/Keras implementation of Deep Image Prior](https://github.com/satoshi-kosugi/DeepImagePrior)
* [Image denoising using deep learning](https://towardsai.net/p/deep-learning/image-de-noising-using-deep-learning)
* [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder/)
* [findpeaks library - denoising](https://erdogant.github.io/findpeaks/pages/html/Denoise.html)
* [Cross correlation](https://en.wikipedia.org/wiki/Cross-correlation)