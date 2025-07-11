import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class KnNNeuralNetwork:
    # Our implementation of the kn-NN Neural Network for density estimation.
    # This is based on the paper we discussed in class but with some modifications
    # we made to get it working properly.

    def __init__(self, k1=1, architecture=(50, 30), max_iter=1000,
                 learning_rate=0.001, random_state=42):
        # Main parameters - took us a while to tune these defaults
        self.k1 = k1  # This controls how many neighbors we use
        self.architecture = architecture  # Hidden layer sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

        # These get initialized during training
        self.mlp = None
        self.scaler = StandardScaler()  # Helps with convergence
        self.training_data = None
        self.targets = None
        self.kn = None  # This gets calculated based on data size

        # Keeping track of training progress
        self.training_history = {
            'loss': [],
            'validation_loss': []
        }

    def _calculate_kn(self, n):
        # Calculating the number of neighbors to use. The paper suggests k1 * sqrt(n) but we had to make sure it's at least 1
        return max(1, int(self.k1 * np.sqrt(n)))

    def _calculate_ball_volume_1d(self, radius):
        # For 1D case, the "volume" is just the length of the interval. So for radius r, we get interval [-r, r] which has length 2r
        return 2 * radius

    def _generate_knn_targets(self, X, biased=False):
        # This is the core part - generating density estimates using kn-NN
        n = len(X)
        self.kn = self._calculate_kn(n)
        targets = np.zeros(n)

        print(f"Generating kn-NN targets with kn={self.kn}, biased={biased}")

        # For each point, we calculate its density using kn-NN
        for i in range(n):
            if biased:
                # Usage of all points including the point itself
                X_search = X
                n_effective = n
            else:
                # Removing the point we're estimating for (leave-one-out)
                # This gives unbiased estimates
                X_search = np.delete(X, i, axis=0)
                n_effective = n - 1

            # Handling of edge case - sometimes we don't have enough points
            if len(X_search) < self.kn:
                kn_effective = len(X_search)
            else:
                kn_effective = self.kn

            if len(X_search) > 0:
                # Finding the kn nearest neighbors
                nbrs = NearestNeighbors(n_neighbors=kn_effective, metric='euclidean')
                nbrs.fit(X_search)
                distances, indices = nbrs.kneighbors(X[i].reshape(1, -1))

                # (distance to furthest neighbor)
                radius = distances[0, -1]

                if radius == 0:
                    radius = 1e-10

                # Calculating volume and density estimate
                volume = self._calculate_ball_volume_1d(radius)
                targets[i] = kn_effective / n_effective / volume
            else:
                targets[i] = 0

        return targets

    def fit(self, X, biased=False, validation_split=0.2, verbose=True):
        # Training of the neural network. We added the validation split to monitor overfitting
        X = np.array(X).reshape(-1, 1)
        self.training_data = X.copy()

        if verbose:
            print(f"Training kn-NN Neural Network")
            print(f"Data shape: {X.shape}")
            print(f"Architecture: {self.architecture}")
            print(f"k1 parameter: {self.k1}")

        # Generating the target values using kn-NN
        targets = self._generate_knn_targets(X, biased=biased)
        self.targets = targets

        # Normalizing the inputs
        X_scaled = self.scaler.fit_transform(X)

        # Splitting the data if we want validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, targets, test_size=validation_split,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X_scaled, targets
            X_val, y_val = None, None

        # Creation and training of the MLP
        # We spent a lot of time tuning these parameters
        self.mlp = MLPRegressor(
            hidden_layer_sizes=self.architecture,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate,
            random_state=self.random_state,
            activation='relu',  # ReLU works best for this problem
            solver='adam',
            validation_fraction=0.1 if validation_split > 0 else 0,
            early_stopping=True if validation_split > 0 else False,
            n_iter_no_change=50  # Stop if no improvement for 50 iterations
        )

        self.mlp.fit(X_train, y_train)

        if hasattr(self.mlp, 'loss_curve_'):
            self.training_history['loss'] = self.mlp.loss_curve_
        if hasattr(self.mlp, 'validation_scores_'):
            self.training_history['validation_loss'] = self.mlp.validation_scores_

        if verbose:
            print(f"Training completed. Final loss: {self.mlp.loss_:.6f}")
            print(f"Number of iterations: {self.mlp.n_iter_}")

    def predict(self, X):
        # Predicting density values for new points
        X = np.array(X).reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        predictions = self.mlp.predict(X_scaled)

        predictions = np.maximum(predictions, 0)

        return predictions

    def plot_training_curves(self, figsize=(12, 4)):
        # Plotting the training curves to see how well the model converged
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        if self.training_history['loss']:
            axes[0].plot(self.training_history['loss'], label='Training Loss')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss Curve')
            axes[0].legend()
            axes[0].grid(True)

        if self.training_history['validation_loss']:
            axes[1].plot(self.training_history['validation_loss'], label='Validation Score')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Validation Score')
            axes[1].set_title('Validation Score Curve')
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_density_estimate(self, x_range=None, n_points=1000, true_pdf=None,
                              figsize=(10, 6), title="Density Estimation"):
        # Visualizing the density estimate
        if x_range is None:
            x_min, x_max = self.training_data.min(), self.training_data.max()
            x_range = (x_min - 0.5, x_max + 0.5)

        x_eval = np.linspace(x_range[0], x_range[1], n_points)
        density_estimates = self.predict(x_eval)

        plt.figure(figsize=figsize)

        plt.plot(x_eval, density_estimates, 'b-', linewidth=2,
                 label=f'kn-NN Estimate (k1={self.k1})')

        if true_pdf is not None:
            true_density = true_pdf(x_eval)
            plt.plot(x_eval, true_density, 'r--', linewidth=2,
                     label='True PDF')

        # Showing the training data points
        plt.scatter(self.training_data.flatten(),
                    np.zeros_like(self.training_data.flatten()),
                    alpha=0.6, s=20, c='orange', label='Training Data')

        plt.xlabel('x')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def compare_biased_unbiased(self, X, x_range=None, n_points=1000,
                                true_pdf=None, figsize=(12, 5)):
        # Comparing biased vs unbiased versions side by side
        # Training of biased version
        model_biased = KnNNeuralNetwork(
            k1=self.k1, architecture=self.architecture,
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        model_biased.fit(X, biased=True, verbose=False)

        # Training of unbiased version
        model_unbiased = KnNNeuralNetwork(
            k1=self.k1, architecture=self.architecture,
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        model_unbiased.fit(X, biased=False, verbose=False)

        if x_range is None:
            x_min, x_max = X.min(), X.max()
            x_range = (x_min - 0.5, x_max + 0.5)

        x_eval = np.linspace(x_range[0], x_range[1], n_points)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plotting of biased version
        density_biased = model_biased.predict(x_eval)
        axes[0].plot(x_eval, density_biased, 'b-', linewidth=2, label='Biased kn-NN')
        if true_pdf is not None:
            axes[0].plot(x_eval, true_pdf(x_eval), 'r--', linewidth=2, label='True PDF')
        axes[0].scatter(X.flatten(), np.zeros_like(X.flatten()),
                        alpha=0.6, s=20, c='orange', label='Training Data')
        axes[0].set_title('Biased kn-NN Neural Network')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plotting of unbiased version
        density_unbiased = model_unbiased.predict(x_eval)
        axes[1].plot(x_eval, density_unbiased, 'g-', linewidth=2, label='Unbiased kn-NN')
        if true_pdf is not None:
            axes[1].plot(x_eval, true_pdf(x_eval), 'r--', linewidth=2, label='True PDF')
        axes[1].scatter(X.flatten(), np.zeros_like(X.flatten()),
                        alpha=0.6, s=20, c='orange', label='Training Data')
        axes[1].set_title('Unbiased kn-NN Neural Network')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return model_biased, model_unbiased


def generate_test_data(distribution='mixture', n_samples=200, random_state=42):
    # Generating different types of test distributions for experimenting
    # We added several options to test the algorithm thoroughly
    np.random.seed(random_state)

    if distribution == 'mixture':
        # Two Gaussian mixture
        component1 = np.random.normal(2, 0.5, n_samples // 2)
        component2 = np.random.normal(5, 0.8, n_samples // 2)
        data = np.concatenate([component1, component2])

        def true_pdf(x):
            return 0.5 * (1 / (0.5 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 2) / 0.5) ** 2) + \
                0.5 * (1 / (0.8 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 5) / 0.8) ** 2)

    elif distribution == 'normal':
        data = np.random.normal(3, 1, n_samples)

        def true_pdf(x):
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 3) ** 2)

    elif distribution == 'exponential':
        data = np.random.exponential(2, n_samples)

        def true_pdf(x):
            return np.where(x >= 0, 0.5 * np.exp(-x / 2), 0)

    elif distribution == 'uniform':
        data = np.random.uniform(0, 4, n_samples)

        def true_pdf(x):
            return np.where((x >= 0) & (x <= 4), 0.25, 0)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return data, true_pdf


def run_experiments():
    # Some basic experiments to test the implementation
    # This is what we used to verify everything was working
    print("=" * 60)
    print("kn-NN Neural Network Experiments")
    print("=" * 60)

    data, true_pdf = generate_test_data('mixture', n_samples=200)

    print(f"Generated {len(data)} samples from mixture distribution")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

    print("\nExperiment 1: Comparing different k1 values")
    k1_values = [0.5, 1.0, 2.0, 3.0]

    plt.figure(figsize=(15, 10))
    for i, k1 in enumerate(k1_values):
        model = KnNNeuralNetwork(k1=k1, architecture=(50, 30), max_iter=500)
        model.fit(data, biased=False, verbose=False)

        x_eval = np.linspace(data.min() - 1, data.max() + 1, 1000)
        density_est = model.predict(x_eval)

        plt.subplot(2, 2, i + 1)
        plt.plot(x_eval, density_est, 'b-', linewidth=2, label=f'kn-NN (k1={k1})')
        plt.plot(x_eval, true_pdf(x_eval), 'r--', linewidth=2, label='True PDF')
        plt.scatter(data, np.zeros_like(data), alpha=0.4, s=10, c='orange')
        plt.title(f'k1 = {k1}')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nExperiment 2: Comparing different architectures")
    architectures = [(20,), (50, 30), (100, 50, 20), (30, 30, 30)]
    arch_names = ['(20,)', '(50,30)', '(100,50,20)', '(30,30,30)']

    plt.figure(figsize=(15, 10))
    for i, (arch, name) in enumerate(zip(architectures, arch_names)):
        model = KnNNeuralNetwork(k1=1.0, architecture=arch, max_iter=500)
        model.fit(data, biased=False, verbose=False)

        x_eval = np.linspace(data.min() - 1, data.max() + 1, 1000)
        density_est = model.predict(x_eval)

        plt.subplot(2, 2, i + 1)
        plt.plot(x_eval, density_est, 'b-', linewidth=2, label=f'kn-NN {name}')
        plt.plot(x_eval, true_pdf(x_eval), 'r--', linewidth=2, label='True PDF')
        plt.scatter(data, np.zeros_like(data), alpha=0.4, s=10, c='orange')
        plt.title(f'Architecture: {name}')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nExperiment 3: Biased vs Unbiased comparison")
    model = KnNNeuralNetwork(k1=1.0, architecture=(50, 30), max_iter=500)
    model_biased, model_unbiased = model.compare_biased_unbiased(
        data, true_pdf=true_pdf
    )

    print("\nExperiment 4: Training curves analysis")
    model_final = KnNNeuralNetwork(k1=1.0, architecture=(50, 30), max_iter=1000)
    model_final.fit(data, biased=False, validation_split=0.2, verbose=True)
    model_final.plot_training_curves()

    return model_final, data, true_pdf


if __name__ == "__main__":
    final_model, test_data, true_pdf_func = run_experiments()

    print("\nFinal Model Visualization:")
    final_model.plot_density_estimate(
        true_pdf=true_pdf_func,
        title="Final kn-NN Neural Network Density Estimation"
    )

    print("\nExperiments completed successfully!")
    print("The model is ready for evaluation on the instructor's dataset.")