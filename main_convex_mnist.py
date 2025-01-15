import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torchvision import datasets, transforms
import os

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def load_mnist_data(data_path):
    """Load MNIST dataset from the given path."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    X = dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y = dataset.targets.numpy()
    return X, y

def generate_convex_iid(X, y, n_samples=100, random_seed=42):
    """Generate a convex IID dataset from MNIST."""
    np.random.seed(random_seed)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]
    y_sample = (y[indices] % 2 == 0).astype(int)  # Binary labels: even=1, odd=0
    return X_sample, y_sample

def generate_convex_non_iid(X, y, n_samples=100, random_seed=42):
    """Generate a convex non-IID dataset from MNIST."""
    np.random.seed(random_seed)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]
    noise = np.random.randn(n_samples) * 0.1
    y_sample = ((y[indices] % 2 == 0) + noise > 0.5).astype(int)  # Add noise to labels
    return X_sample, y_sample

def generate_non_convex_iid(X, y, n_samples=100, random_seed=42):
    """Generate a non-convex IID dataset from MNIST."""
    np.random.seed(random_seed)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]
    y_sample = (np.sin(np.sum(X_sample, axis=1)) > 0).astype(int)  # Non-linear function
    return X_sample, y_sample

def generate_non_convex_non_iid(X, y, n_samples=100, random_seed=42):
    """Generate a non-convex non-IID dataset from MNIST."""
    np.random.seed(random_seed)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]
    noise = np.random.randn(n_samples) * 0.1
    y_sample = (np.sin(np.sum(X_sample, axis=1)) + noise > 0).astype(int)  # Add noise to non-linear function
    return X_sample, y_sample

# Path to MNIST dataset on GitHub
data_path = os.path.join("dataset", "MNIST", "MNIST", "raw")
X, y = load_mnist_data(data_path)

# Generate datasets
datasets = {
    "convex_iid": generate_convex_iid(X, y),
    "convex_non_iid": generate_convex_non_iid(X, y),
    "non_convex_iid": generate_non_convex_iid(X, y),
    "non_convex_non_iid": generate_non_convex_non_iid(X, y),
}

# Visualize samples from datasets
for name, (X_sample, y_sample) in datasets.items():
    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(y_sample)), y_sample, c=y_sample, cmap="coolwarm", edgecolor="k", s=10)
    plt.title(f"Dataset: {name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Label")
    plt.grid(True)
    plt.show() 
