# PyTorch Retrieve 🛰️

Neural-network-based remote-sensing retrievals for the busy scientist. 👩‍🔬

## Why PyTorch Retrieve?

> You provide the data, PyTorch Retrieve does the rest

PyTorch Retrieve aims to  simplify the retrieval of geophysical quantities from satellite observations. Its principal features are:

    1. Powerful and flexible neural network architectures for quantification and classification.
    2. Handling of the aleatoric uncertainty caused by the inverse-problem character of the retrieval using probabilistic regression.
    3. Tools to bring the retrieval from experiment to production
    


## PyTorch Retrieve vs. other packages for geo-spatial DL

Given that there's already [TorchGeo](https://github.com/microsoft/torchgeo) and
[TorchSat](https://github.com/sshuair/torchsat), why another deep-learning
package for satellite data?

Neither TorchGeo nor TorchSat were designed with quantification tasks in mind
and thus do not support dense quantification tasks or quantification of prediction uncertainies. Moreover, both packages are designed to make it easier to load geospatial data into PyTorch.

PyTorch Retrieve, on the other hand, works in the different direction: Instead of helping you bring your data into PyTorch, it aims to help you get PyTorch boiler-plate code out of your retrieval. It aims to eliminate all the nasty, hard-coded PyTorch details from your application thus allowing you to modify or switch out the underlying
 neural-network architecture.

## PyTorch Retrieve vs. PyTorch Lightning

PyTorch Lightning is straight-up amazing. However, building a retrieval based on
it still requires writing a large amount of mostly boiler-plate PyTorch code.


## Limitations

as other projects 
Satellite remote sensing (RS) is full of probabilistic regression problems with small but significant differences to the more common computer vision (CV) applications. Examples are:

  1. The input data has more than 3 channels: Very few (if any) satellite sensors measure only red, green, and blue bands. This precludes most existing pre-trained CV models.
  2. Uncertainties matter: Many remote sensing tasks are ultimately inverse problems
    thus incurring non-negligible prediction uncertainties and quantifying them is desired for applications built on these predictions.
  3. While lightning significantly simplifies many of the technical aspects of training deep neural networks, building applications on it still requires significnat amounts of boiler plate code. Moreover, hard-coding network architecture and hyper-parameters is likely to lead to messy code and reproducibility issues.
  4. Finally, resolution often matters: Dense predictions in many RS applications need to be performed at the same resolution as the input data, which, again is in contrast to most dense predictions tasks in computer vision.

``pytorch_retrieve`` addresses these shortcomings by 

 1. Providing a clean and easy-to-use library to perform probabilistic regression on tabular (single pixel + 1 spectral dimension), spatial (2 spatial dimensions + 1 spectral dimension) and spatio-temporal (1 time dimension + 2 spatial dimension + 1 spectral dimension) multi-spectral remote sensing data,
 2. introducing an abstraction layer between the remote sensing task and the underlying network architecture,
 3. providing functionality to reduce boiler plate code and ease the transition from experimental to operational application.
