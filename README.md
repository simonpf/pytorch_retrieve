# PROBREG 📈

Probabilistic regression (and more) for the busy remote-sensing scientist. 👩‍🔬


## Why PROBREG?

Satellite remote sensing (RS) is full of probabilistic regression problems with small but significant differences to the more common computer vision (CV) applications. Examples are:

  1. The input data has more than 3 channels: Very few (if any) satellite sensors measure only red, green, and blue bands. This precludes most existing pre-trained CV models.
  2. Uncertainties matter: Many remote sensing tasks are ultimately inverse problems
    thus incurring non-negligible prediction uncertainties and quantifying them is desired for applications built on these predictions.
  3. While lightning significantly simplifies many of the technical aspects of training deep neural networks, building applications on it still requires significnat amounts of boiler plate code. Moreover, hard-coding network architecture and hyper-parameters is likely to lead to messy code and reproducibility issues.
  4. Finally, resolution often matters: Dense predictions in many RS applications need to be performed at the same resolution as the input data, which, again is in contrast to most dense predictions tasks in computer vision.

PROBREG addresses these shortcomings by 

 1. Providing a clean and easy-to-use library to perform probabilistic regression on tabular (single pixel + 1 spectral dimension), spatial (2 spatial dimensions + 1 spectral dimension) and spatio-temporal (1 time dimension + 2 spatial dimension + 1 spectral dimension) multi-spectral remote sensing data,
 2. introducing an abstraction layer between the remote sensing task and the underlying network architecture,
 3. providing functionality to reduce boiler plate code and ease the transition from experimental to operational application.
