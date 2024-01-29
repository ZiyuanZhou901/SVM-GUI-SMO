import tkinter as tk
from tkinter import ttk
from smo_solver import SVM
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
import time
import sys

class SVMGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("SVM")

        self.master.geometry("1200x550")
        self.master.resizable(False, False)
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)

        self.dimension_var = tk.StringVar(value='3')
        self.data_size_var = tk.StringVar(value='1000')
        self.generate_method_var = tk.StringVar(value='random')

        self.solve_button = ttk.Button(self.master, text="Generate Data & Solve", command=self.generate_data_and_solve)
        self.solve_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.matplotlib_frame = ttk.Frame(self.master)
        self.matplotlib_frame.grid(row=5, column=0, columnspan=2, pady=10)

        self.create_matplotlib_frame()

        self.create_widgets()

    def create_matplotlib_frame(self):
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.matplotlib_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def create_widgets(self):
        dimension_label = ttk.Label(self.master, text="Select Dimension:")
        dimension_combobox = ttk.Combobox(self.master, textvariable=self.dimension_var, values=['3', '4', '5', '6'], state='readonly')

        data_size_label = ttk.Label(self.master, text="Select Data Size:")
        data_size_combobox = ttk.Combobox(self.master, textvariable=self.data_size_var, values=['1000','10000', '100000', '1000000'], state='readonly')

        solve_button = ttk.Button(self.master, text="Generate Data & Solve", command=self.generate_data_and_solve)

        results_label = ttk.Label(self.master, text="Results:")
        self.results_text = tk.Text(self.master, height=10, width=50, state='disabled')

        method_label = ttk.Label(self.master, text="Generation Method:")
        random_button = ttk.Radiobutton(self.master, text="Random", variable=self.generate_method_var, value='random', command=self.toggle_entry)
        manual_button = ttk.Radiobutton(self.master, text="Manual", variable=self.generate_method_var, value='manual', command=self.toggle_entry)

        self.w_label = ttk.Label(self.master, text="Enter w (eg: 1,2,3):")
        self.w_entry = ttk.Entry(self.master)
        self.b_label = ttk.Label(self.master, text="Enter b:")
        self.b_entry = ttk.Entry(self.master)

        # Layout
        dimension_label.grid(row=0, column=0, padx=10, pady=0, sticky=tk.W)
        dimension_combobox.grid(row=0, column=1, padx=10, pady=0, sticky=tk.E)
        data_size_label.grid(row=1, column=0, padx=10, pady=0, sticky=tk.W)
        data_size_combobox.grid(row=1, column=1, padx=10, pady=0, sticky=tk.E)
        solve_button.grid(row=2, column=0, columnspan=2, pady=0)
        results_label.grid(row=3, column=0, columnspan=2, pady=0)
        self.matplotlib_frame.grid(row=0, column=2, rowspan=5, pady=0, padx=10, sticky=tk.E)
        self.results_text.grid(row=4, column=0, columnspan=2, pady=0, sticky=tk.W+tk.E)
        method_label.grid(row=5, column=1, pady=10, sticky=tk.W)
        random_button.grid(row=6, column=0, pady=5, sticky=tk.W)
        manual_button.grid(row=6, column=1, pady=5, sticky=tk.W)

    def toggle_entry(self):
        if self.generate_method_var.get() == 'manual':
            self.w_label.grid(row=5, column=2, pady=5, sticky=tk.W)
            self.w_entry.grid(row=5, column=3, pady=5, sticky=tk.W)
            self.b_label.grid(row=6, column=2, pady=5, sticky=tk.W)
            self.b_entry.grid(row=6, column=3, pady=5, sticky=tk.W)
        else:
            self.w_label.grid_forget()
            self.w_entry.grid_forget()
            self.b_label.grid_forget()
            self.b_entry.grid_forget()

    def generate_data_and_solve(self):
        dimension = int(self.dimension_var.get())
        data_size = int(self.data_size_var.get())

        if self.generate_method_var.get() == 'random':
            X, y, w_true, b_true = generate_data(data_size, dimension)
        else:
            try:
                w_input = np.array([float(x) for x in self.w_entry.get().split(',')])
                b_input = float(self.b_entry.get())

                if len(w_input) != dimension:
                    raise ValueError("Length of w should be equal to the selected dimension.")

            except ValueError as e:
                tk.messagebox.showerror("Error", f"Invalid input: {str(e)}")
                return
            X, y, w_true, b_true = generate_data_manual(data_size, dimension, w_input, b_input)
        
        time1 = time.time()
        model = SVM(kkt_thr=1e-8)
        model.fit(X, y)
        time2 = time.time()
        w = np.dot(model.alpha * model.support_labels, model.support_vectors)
        b = np.mean(model.support_labels - np.dot(model.support_vectors, w))
        Time = time2 - time1

        results_text = f"True Hyperplane:\nw_true: {w_true}\nb_true: {b_true}\n\n"
        results_text += f"Optimal w: {w}\n"
        results_text += f"Optimal b: {b}\n"
        results_text += f"Time spent: {Time}\n"

        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)
        self.results_text.config(state='disabled')

        self.ax.clear()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        w_optimal_pca = pca.transform(w.reshape(1, -1))
        self.plot_hyperplane(X_pca, y, w_optimal_pca[0], b, title='Optimal Hyperplane')

        self.canvas.draw()

        self.solve_button.config(state='normal')


    def plot_hyperplane(self, X, y, w, b, title):
        self.ax.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', marker='o')
        self.ax.scatter(X[y == -1, 0], X[y == -1, 1], label='Class -1', marker='x')
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
        Z = Z.reshape(xx.shape)
        self.ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        self.ax.set_title(title)
        self.ax.set_xlabel('Feature 1')
        self.ax.set_ylabel('Feature 2')
        self.ax.legend()

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

def generate_data_manual(N,n,w_input,b_input):
    sign_x = np.random.choice([-1, 1], size=(N, n))
    X = sign_x * np.random.rand(N, n)*30
    w_true = w_input
    b_true = b_input
    print("True Hyperplane:")
    print("w_true:", w_true)
    print("b_true:", b_true)
    y = np.sign(np.dot(X, w_true) + b_true)
    return X, y, w_true, b_true

if __name__ == "__main__":
    root = tk.Tk()
    app = SVMGUI(root)
    root.protocol("WM_DELETE_WINDOW",lambda: sys.exit(0))
    root.mainloop()

