# PyTorch Retrieve 🛰️

Neural-network-based remote-sensing retrievals for the busy scientist. 👩‍🔬

## Why PyTorch Retrieve?

The aim of PyTorch Retrieve is to provide remote-sensing scientists with a fast,
fail-safe and flexible way of training retrievals of geophysical quantities from
satellite observations. PyTorch principal features of PyTorch retrieval are the following:

1. Flexible implementations of state-of-the-art neural network architecture that
   can be trained on a wide range of input data including multi-spectral,
   multi-sensor and multi time step data.
2. Multi-output retrievals handling scalar, vector, continuous and catergorical outputs. 
3. Modular model configuration using configuration files in '.toml' or '.yaml' format.
4. Probabilistic regression using quantiles or binned distributions.

## PyTorch Retrieve vs. other packages for geo-spatial DL

Why another deep-learning package for satellite data?

The other deep-learning pacakges for geospatial data that I am aware of  ([TorchGeo](https://github.com/microsoft/torchgeo) and [TorchSat](https://github.com/sshuair/torchsat)) were designed with classification tasks in mind and most of their functionality focuses on loading geospatial data or providing interfaces to existing geospatial ML datasets.

PyTorch Retrieve takes the opposite approach in the sense that leaves the preparation and loading of the data completely to the user and instead takes care of all the machine-learning-related details. PyTorch Retrieve aims separate the science code (the preparation of the training data and evaluation of the retrieval) from the engineering (the training of the machine learning model). Separating concerns in this way makes it easy to modify or completely swith-out the neural network architecture used for the retrieval.
