"""Test script based on the README examples to verify the refactored code."""
import numpy as np
from river.datasets import synth

# --- Generate dataset ---
print("Generating dataset...")
ds = synth.RandomRBF(seed_model=42, seed_sample=42, n_classes=4, n_features=5, n_centroids=10)

offline_size = 1000
online_size = 5000
X_train = []
y_train = []
X_test = []
y_test = []

for x, y in ds.take(10 * (offline_size + online_size)):
    if len(y_train) < offline_size:
        if y == 0 or y == 1:
            X_train.append(np.array(list(x.values())))
            y_train.append(y)
    elif len(y_test) < online_size:
        X_test.append(x)
        y_test.append(y)
    else:
        break

X_train = np.array(X_train)
y_train = np.array(y_train)

known_classes = [0, 1]


# print("\n=== Testing MINAS ===")
from streamndr.model import Minas
from streamndr.metrics import ConfusionMatrixNovelty, MNew, FNew, ErrRate

# clf = Minas(kini=100, cluster_algorithm='kmeans',
#             window_size=600, threshold_strategy=1, threshold_factor=1.1,
#             min_short_mem_trigger=100, min_examples_cluster=20, verbose=0, random_state=42)

# clf.learn_many(np.array(X_train), np.array(y_train))

# conf_matrix = ConfusionMatrixNovelty(known_classes)
# m_new = MNew(known_classes)
# f_new = FNew(known_classes)
# err_rate = ErrRate(known_classes)

# for i, (x, y_true) in enumerate(zip(X_test, y_test), 1):
#     y_pred = clf.predict_one(x)
#     if y_pred is not None:
#         conf_matrix.update(y_true, y_pred[0])
#         m_new.update(y_true, y_pred[0])
#         f_new.update(y_true, y_pred[0])
#         err_rate.update(y_true, y_pred[0])
#     if i % 1000 == 0:
#         print(f"  {i}/{len(X_test)}")

# print(m_new)
# print(f_new)
# print(err_rate)


print("\n=== Testing ECSMiner-WF ===")
from streamndr.model import ECSMinerWF

clf2 = ECSMinerWF(K=50, min_examples_cluster=10, verbose=0, random_state=42, ensemble_size=7, init_algorithm="kmeans")
clf2.learn_many(np.array(X_train), np.array(y_train))

conf_matrix2 = ConfusionMatrixNovelty(known_classes)
m_new2 = MNew(known_classes)
f_new2 = FNew(known_classes)
err_rate2 = ErrRate(known_classes)

for i, (x, y_true) in enumerate(zip(X_test, y_test), 1):
    y_pred = clf2.predict_one(x)
    if y_pred is not None:
        conf_matrix2.update(y_true, y_pred[0])
        m_new2.update(y_true, y_pred[0])
        f_new2.update(y_true, y_pred[0])
        err_rate2.update(y_true, y_pred[0])
    if i % 1000 == 0:
        print(f"  {i}/{len(X_test)}")

print(m_new2)
print(f_new2)
print(err_rate2)