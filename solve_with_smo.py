import numpy as np
from smo_solver import SVM
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_hyperplane(X, y, w, b, title):
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', marker='o')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='Class -1', marker='x')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def generate_data(N, n):
    sign_x = np.random.choice([-1, 1], size=(N, n))
    X = sign_x * np.random.rand(N, n)*30
    sign_w = np.random.choice([-1, 1], size=n)
    sign_b = np.random.choice([-1, 1])
    w_true = sign_w * np.random.rand(n)
    b_true = sign_b * np.random.rand()
    print("True Hyperplane:")
    print("w_true:", w_true)
    print("b_true:", b_true)
    y = np.sign(np.dot(X, w_true) + b_true)
    return X, y, w_true, b_true

n=3
N=10000

X,y,w_true,b_true=generate_data(N,n)
model=SVM(kkt_thr=1e-8)
start_time = time.time()
model.fit(X,y)
end_time = time.time()
time_spent = end_time - start_time
print("Time spent:", time_spent)

w=np.dot(model.alpha*model.support_labels,model.support_vectors)
b=np.mean(model.support_labels-np.dot(model.support_vectors,w))

print("optimal_w:", w)
print("optimal_b:", b)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
w_optimal_pca = pca.transform(w.reshape(1, -1))
plot_hyperplane(X_pca, y, w_optimal_pca[0], b, title='Optimal Hyperplane')
