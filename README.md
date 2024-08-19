<p align="center">
<!-- Documentation -->
  <a href="https://jgaud.github.io/streamndr/">
    <img src="https://img.shields.io/website?label=docs&style=flat-square&url=https%3A%2F%2Fjgaud.github.io%2Fstreamndr%2F" alt="documentation">
  </a>
<!-- PyPI -->
  <a href="https://pypi.org/project/streamndr/">
    <img src="https://img.shields.io/pypi/v/streamndr.svg?label=release&color=blue&style=flat-square" alt="pypi">
  </a>
  <!-- License -->
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="bsd_3_license">
  </a>
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/streamndr">
</p>
<p align="center">
    Stream Novelty Detection for River (StreamNDR) is a Python library for online novelty detection.
    StreamNDR aims to enable <a href="https://deepai.org/machine-learning-glossary-and-terms/novelty-detection">novelty detection</a> in data streams for Python.
    It is based on the <a href="https://www.riverml.xyz">river</a> API and is currently in early stage of development. Contributors are welcome.
</p>

## 📚 [Documentation](https://jgaud.github.io/streamndr/)
StreamNDR implements in Python various algorithms for novelty detection that have been proposed in the literature. It follows <a href="https://www.riverml.xyz">river</a> implementation and format. At this stage, the following algorithms are implemented:
- MINAS [[1]](#1)
- ECSMiner [[2]](#2)
- ECSMiner-WF (Version of ECSMiner [[2]](#2) without feedback, as proposed in [[1]](#1))

Full documentation is available [here](https://jgaud.github.io/streamndr/).

## 🛠 Installation
**Note**: StreamNDR is intended to be used with Python 3.6 or above and requires the package [ClusOpt-Core](https://pypi.org/project/clusopt-core/) which requires a C/C++ compiler (such as gcc) and the [Boost.Thread library](https://robots.uc3m.es/installation-guides/install-boost.html) to build. To install the Boost.Thread library on Debian systems, the following command can be used:

```console
sudo apt install libboost-thread-dev
```

The package can be installed simply with `pip` :
```console
pip install streamndr
```

## ⚡️ Quickstart

As a quick example, we'll train two models (MINAS and ECSMiner-WF) to classify a synthetic dataset created using [RandomRBF](https://riverml.xyz/dev/api/datasets/synth/RandomRBF/). The models are trained on only two of the four generated classes ([0,1]) and will try to detect the other classes ([2,3]) as novelty patterns in the dataset in an online fashion.

Let's first generate the dataset.
```python
import numpy as np
from river.datasets import synth

ds = synth.RandomRBF(seed_model=42, seed_sample=42, n_classes=4, n_features=5, n_centroids=10)

offline_size = 1000
online_size = 5000
X_train = []
y_train = []
X_test = []
y_test = []

for x,y in ds.take(10*(offline_size+online_size)):
    
    #Create our training data (known classes)
    if len(y_train) < offline_size:
        if y == 0 or y == 1: #Only showing two first classes in the training set
            X_train.append(np.array(list(x.values())))
            y_train.append(y)
    
    #Create our online stream of data
    elif len(y_test) < online_size:
        X_test.append(x)
        y_test.append(y)
        
    else:
        break

X_train = np.array(X_train)
y_train = np.array(y_train)
```


### MINAS
Let's train our MINAS model on the offline (known) data.
```python
from streamndr.model import Minas
clf = Minas(kini=100, cluster_algorithm='clustream', 
            window_size=600, threshold_strategy=1, threshold_factor=1.1, 
            min_short_mem_trigger=100, min_examples_cluster=20, verbose=1, random_state=42)

clf.learn_many(np.array(X_train), np.array(y_train)) #learn_many expects numpy arrays or pandas dataframes
```

Let's now test our algorithm in an online fashion, note that our unsupervised clusters are automatically updated with the call to ```predict_one```.

```python
from streamndr.metrics import ConfusionMatrixNovelty, MNew, FNew, ErrRate

known_classes = [0,1]

conf_matrix = ConfusionMatrixNovelty(known_classes)
m_new = MNew(known_classes)
f_new = FNew(known_classes)
err_rate = ErrRate(known_classes)

i = 1
for x, y_true in zip(X_test, y_test):

    y_pred = clf.predict_one(x) #predict_one takes python dictionaries as per River API
    
    if y_pred is not None: #Update our metrics
        conf_matrix.update(y_true, y_pred[0])
        m_new.update(y_true, y_pred[0])
        f_new.update(y_true, y_pred[0])
        err_rate.update(y_true, y_pred[0])


    #Show progress
    if i % 100 == 0:
        print(f"{i}/{len(X_test)}")
    i += 1
```

Let's look at the results, of course, the hyperparameters of the model can be tuned to get better results.

```python
#print(conf_matrix) #Shows the confusion matrix of the given problem, can be very wide due to one class being detected as multiple Novelty Patterns
print(m_new) #Percentage of novel class instances misclassified as known.
print(f_new) #Percentage of known classes misclassified as novel.
print(err_rate) #Total misclassification error percentage
```
MNew: 17.15% <br/>
FNew: 40.11% <br/>
ErrRate: 36.80% <br/>


### ECSMiner-WF
Let's train our model on the offline (known) data.

```python
from streamndr.model import ECSMinerWF
clf = ECSMinerWF(K=50, min_examples_cluster=10, verbose=1, random_state=42, ensemble_size=7, init_algorithm="kmeans")
clf.learn_many(np.array(X_train), np.array(y_train))
```
Once again, let's use our model in an online fashion.
```python
conf_matrix = ConfusionMatrixNovelty(known_classes)
m_new = MNew(known_classes)
f_new = FNew(known_classes)
err_rate = ErrRate(known_classes)

for x, y_true in zip(X_test, y_test):

    y_pred = clf.predict_one(x) #predict_one takes python dictionaries as per River API

    if y_pred is not None: #Update our metrics
        conf_matrix.update(y_true, y_pred[0])
        m_new.update(y_true, y_pred[0])
        f_new.update(y_true, y_pred[0])
        err_rate.update(y_true, y_pred[0])
```

```python
#print(conf_matrix) #Shows the confusion matrix of the given problem, can be very wide due to one class being detected as multiple Novelty Patterns
print(m_new) #Percentage of novel class instances misclassified as known.
print(f_new) #Percentage of known classes misclassified as novel.
print(err_rate) #Total misclassification error percentage
```

MNew: 28.98% <br/>
FNew: 30.26% <br/>
ErrRate: 32.40% <br/>

## Special Thanks
Special thanks goes to Vítor Bernardes, from which some of the code for MINAS is based on their [implementation](https://github.com/vbernardes/minas).

## 💬 References
<a id="1">[1]</a> 
de Faria, E.R., Ponce de Leon Ferreira Carvalho, A.C. & Gama, J. MINAS: multiclass learning algorithm for novelty detection in data streams. Data Min Knowl Disc 30, 640–680 (2016). https://doi.org/10.1007/s10618-015-0433-y

<a id="2">[2]</a>
M. Masud, J. Gao, L. Khan, J. Han and B. M. Thuraisingham, "Classification and Novel Class Detection in Concept-Drifting Data Streams under Time Constraints," in IEEE Transactions on Knowledge and Data Engineering, vol. 23, no. 6, pp. 859-874, June 2011, doi: 10.1109/TKDE.2010.61.

## 🏫 Affiliations

<p align="center">
    <img src="http://www.uottawa.ca/brand/sites/www.uottawa.ca.brand/files/uottawa_hor_wg9.png" alt="FZI Logo" height="200"/>
</p>
