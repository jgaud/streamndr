# StreamNDR

This site contains the documentation for `StreamNDR`, a Python library based on [river](https://riverml.xyz) which aims to implement novelty detection algorithm for data streams. The library is open-source and availabble on [Github](https://github.com/jgaud/streamndr).

## What is novelty detection?

Novelty Detection consists of the task of detecting novelty concepts (or classes) in a data stream, that is, classes that the model has not seen before. In order to do so, the algorithms often implement an offline phase, where ther learn the known classes in supervised manner, followed by an online phase, where the algorithm will try to label and detect novel classes within the stream of data. 