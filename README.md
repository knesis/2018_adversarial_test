## Wavelet Denoising as Defense for Adversarial Attacks

These Matlab files produce figures demonstrating the use of wavelet thresholding for removing adversarial image distortion.

Adversarial images were generated in Python Tensorflow using the approach of Anish Athalye for L-inf norm distortion: 
https://www.anishathalye.com/2017/07/25/synthesizing-adversarial-examples.
Data set includes both adversarial and "robust" adversarial (rotationally invariant) images.

Pretrained Inceptionv3 image classification network was used to evaluate performance of wavelet thresholding.
Pre-distorted images provided in /Images folder for both original and non-robust/robust altered images. 

Native Matlab wavelet toolbox with biorthogonal 3.5 wavelet utilized for denoising.

Presentation for viability of method provided in "adversarial_denoising.pptx."
