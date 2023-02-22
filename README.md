<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/jgaud/streamndr">
</p>
<p align="center">
    Stream Novelty Detection for River (StreamNDR) is a Python library for online novelty detection.
    StreamNDR aims to enable <a href="https://deepai.org/machine-learning-glossary-and-terms/novelty-detection">novelty detection</a> in data streams for Python.
    It is based on the <a href="https://www.riverml.xyz">river</a> API and is currently in early stage of development. Contributors are welcome.
</p>

## 📚 [Documentation]()
StreamNDR implements in Python various algorithms for novelty detection that have been proposed in the literature. It follows <a href="https://www.riverml.xyz">river</a> implementation and format. At this stage, the following algorithms are implemented:
- MINAS [[1]](#1)
- ECSMiner-WF (Version of ECSMiner [[2]](#2) without feedback, as proposed in [[1]](#1))

Full documentation is available at: To do.

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
clf = Minas(kini=10, cluster_algorithm='kmeans', 
            window_size=100, threshold_strategy=1, threshold_factor=1.1, 
            min_short_mem_trigger=100, min_examples_cluster=3, verbose=1, random_state=42)

clf.learn_many(np.array(X_train), np.array(y_train)) #learn_many expects numpy arrays or pandas dataframes
```

Let's now test our algorithm in an online fashion, note that our unsupervised clusters are automatically updated with the call to ```predict_one```.

```python
from river import metrics

conf_matrix = metrics.ConfusionMatrix()

for x, y_true in zip(X_test, y_test):

    y_pred = clf.predict_one(x) #predict_one takes python dictionaries as per River API
    
    if y_pred is not None:
        conf_matrix = conf_matrix.update(y_true, y_pred[0])
```
Looking at the confusion matrix below, with -1 being the unknown class, we can see that our model succesfully detected some of our novel classes ([3,4]) as novel concepts. Of course, the hyperparameters of the model can be tuned a lot more to get better results.
```python
print(conf_matrix)
```
|        | **-1** | **0** | **1** | **2** | **3** |
|--------|--------|-------|-------|-------|-------|
| **-1** | 0      | 0     | 0     | 0     | 0     |
| **0**  | 819    | 286   | 17    | 6     | 22    |
| **1**  | 1260   | 3     | 1182  | 84    | 3     |
| **2**  | 372    | 2     | 14    | 336   | 0     |
| **3**  | 137    | 0     | 0     | 0     | 457   |

### ECSMiner-WF
Let's train our model on the offline (known) data.

```python
clf = ECSMinerWF(K=5, min_examples_cluster=5, verbose=1, random_state=42, ensemble_size=20)
clf.learn_many(np.array(X_train), np.array(y_train))
```
The confusion matrix shows us that ECSMiner successfully detected our third class as novel concepts, but not our second class. Again, a lot more tuning can be done to the hyperparameters to improve the results. It is to be noted too that ECSMiner is originally an algorithm that receives feedback (true values) back from the user. With feedback, the algorithm would perform a lot better.
print(conf_matrix)
```
Once again, let's test our model in an online fashion.
```python
conf_matrix = metrics.ConfusionMatrix()

for x, y_true in zip(X_test, y_test):

    y_pred = clf.predict_one(x) #predict_one takes python dictionaries as per River API
    
    if y_pred is not None:
        conf_matrix = conf_matrix.update(y_true, y_pred[0])
```



|        | **-1** | **0** | **1** | **2** | **3** | **4** | **5** | **6** |
|--------|--------|-------|-------|-------|-------|-------|-------|-------|
| **-1** | 0      | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
| **0**  | 92     | 835   | 219   | 3     | 0     | 0     | 1     | 0     |
| **1**  | 216    | 180   | 2131  | 0     | 0     | 1     | 2     | 2     |
| **2**  | 44     | 6     | 673   | 0     | 0     | 1     | 0     | 0     |
| **3**  | 106    | 280   | 88    | 0     | 67    | 23    | 19    | 11    |
| **4**  | 0      | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
| **5**  | 0      | 0     | 0     | 0     | 0     | 0     | 0     | 0     |
| **6**  | 0      | 0     | 0     | 0     | 0     | 0     | 0     | 0     |

## 🛠 Installation

To Do.

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
