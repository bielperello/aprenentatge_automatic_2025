import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from Perceptron import Perceptron

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.25,
                           random_state=0)

y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.


perceptron = Perceptron()
perceptron.fit(X, y)  # Ajusta els pesos
y_prediction = perceptron.predict(X)  # Prediu

# Resultats
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_prediction, cmap="bwr", alpha=0.7)

# Dibuixar la recta de decisió
w0, w1, w2 = perceptron.w_
slope = -w1 / w2
intercept = -w0 / w2  # punt on talla l’eix Y
ax.axline((0, intercept), slope=slope, color="black", linestyle="--", linewidth=2)

ax.set_title("Separació trobada pel Perceptron")
plt.show()

