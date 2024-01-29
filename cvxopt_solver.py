import cvxopt
import cvxopt.solvers
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def solve_optimal_hyperplane_dual(X, y, tolerance=1e-5):
    m, n = X.shape

    K = np.dot(X, X.T)
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-np.ones((m, 1)))
    G = cvxopt.matrix(-np.eye(m))
    h = cvxopt.matrix(np.zeros(m))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))

    start_time = time.time()
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    end_time = time.time()
    time_spent = end_time - start_time

    alpha = np.array(solution['x'])

    sv_indices = (alpha > tolerance).flatten()
    w = np.sum(alpha * y.reshape(-1, 1) * X, axis=0)
    b = np.mean(y[sv_indices] - np.dot(X[sv_indices], w))
    return w, b, time_spent

def generate_data(N, n):
    sign_x = np.random.choice([-1, 1], size=(N, n))
    X = sign_x * np.random.rand(N, n)*20
    sign_w = np.random.choice([-1, 1], size=n)
    sign_b = np.random.choice([-1, 1])
    w_true = sign_w * np.random.rand(n)
    b_true = sign_b * np.random.rand()
    print("True Hyperplane:")
    print("w_true:", w_true)
    print("b_true:", b_true)
    y = np.sign(np.dot(X, w_true) + b_true)
    return X, y, w_true, b_true

def plot_data(X, y, title):
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', marker='o')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='Class -1', marker='x')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def plot_hyperplane(X, y, w, b, title):
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', marker='o')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label='Class -1', marker='x')
    
    # Plot the hyperplane
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

def main():
    
    N = 1000
    n = 3
    X, y, true_w, true_b = generate_data(N, n)

    w_optimal, b_optimal, time_spent = solve_optimal_hyperplane_dual(X, y)

    print("Optimal Hyperplane (Dual Problem):")
    print("w_optimal:", w_optimal)
    print("b_optimal:", b_optimal)
    print("Time spent:", time_spent)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    w_optimal_pca = pca.transform(w_optimal.reshape(1, -1))
    plot_hyperplane(X_pca, y, w_optimal_pca[0], b_optimal, title='Optimal Hyperplane (Dual Problem)')

if __name__ == "__main__":
    main()