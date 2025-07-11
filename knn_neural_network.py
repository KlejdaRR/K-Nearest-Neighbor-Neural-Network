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

    def __init__(self, k1=1.0, architecture=(100, 50), max_iter=2000,
                 learning_rate=0.0001, random_state=42, alpha=0.001):
        self.k1 = k1
        self.architecture = architecture
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.alpha = alpha  # added L2 regularization

        self.mlp = None
        self.scaler = StandardScaler()
        self.training_data = None
        self.targets = None
        self.kn = None
        self.data_range = None

        self.training_history = {
            'loss': [],
            'validation_loss': []
        }

    def _calculate_kn(self, n):
        kn = max(1, int(self.k1 * np.sqrt(n)))
        # Cap at reasonable fraction of dataset
        kn = min(kn, n // 3)
        return kn

    def _calculate_ball_volume_1d(self, radius):
        # 1D ball volume with minimum threshold
        return max(2 * radius, 1e-8)

    def _generate_knn_targets(self, X, biased=False):
        # Improved target generation with better stability
        n = len(X)
        self.kn = self._calculate_kn(n)
        targets = np.zeros(n)

        print(f"Generating kn-NN targets with kn={self.kn}, biased={biased}")

        # Stored data range for boundary handling
        self.data_range = (X.min(), X.max())
        data_span = self.data_range[1] - self.data_range[0]

        for i in range(n):
            if biased:
                X_search = X
                n_effective = n
            else:
                X_search = np.delete(X, i, axis=0)
                n_effective = n - 1

            if len(X_search) < self.kn:
                kn_effective = len(X_search)
            else:
                kn_effective = self.kn

            if len(X_search) > 0:
                nbrs = NearestNeighbors(n_neighbors=kn_effective, metric='euclidean')
                nbrs.fit(X_search)
                distances, indices = nbrs.kneighbors(X[i].reshape(1, -1))

                radius = distances[0, -1]

                if radius == 0:
                    radius = data_span / (10 * n)

                # Added small regularization to prevent extreme values
                radius = max(radius, data_span / (100 * n))

                volume = self._calculate_ball_volume_1d(radius)
                density_estimate = kn_effective / n_effective / volume

                max_reasonable_density = 10 / data_span
                targets[i] = min(density_estimate, max_reasonable_density)
            else:
                targets[i] = 0

        return targets

    def fit(self, X, biased=False, validation_split=0.2, verbose=True):
        X = np.array(X).reshape(-1, 1)
        self.training_data = X.copy()

        if verbose:
            print(f"Training Improved kn-NN Neural Network")
            print(f"Data shape: {X.shape}")
            print(f"Architecture: {self.architecture}")
            print(f"k1 parameter: {self.k1}")

        targets = self._generate_knn_targets(X, biased=biased)
        self.targets = targets

        # Normalizaztion of targets
        target_scale = np.std(targets)
        if target_scale > 0:
            targets = targets / target_scale
            self.target_scale = target_scale
        else:
            self.target_scale = 1.0

        X_scaled = self.scaler.fit_transform(X)

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, targets, test_size=validation_split,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X_scaled, targets
            X_val, y_val = None, None

        # Improved MLP with better regularization
        self.mlp = MLPRegressor(
            hidden_layer_sizes=self.architecture,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate,
            random_state=self.random_state,
            activation='relu',
            solver='adam',
            alpha=self.alpha,  # L2 regularization
            validation_fraction=0.1 if validation_split > 0 else 0,
            early_stopping=True if validation_split > 0 else False,
            n_iter_no_change=100,  # More patience
            tol=1e-6  # Tighter tolerance
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
        X = np.array(X).reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        predictions = self.mlp.predict(X_scaled)

        predictions = predictions * self.target_scale

        predictions = np.maximum(predictions, 0)

        # Here we have applied boundary damping to prevent extreme values at edges
        if self.data_range is not None:
            data_min, data_max = self.data_range
            data_span = data_max - data_min
            boundary_buffer = 0.1 * data_span

            outside_mask = (X.flatten() < data_min - boundary_buffer) | \
                           (X.flatten() > data_max + boundary_buffer)

            if np.any(outside_mask):
                # Exponential decay outside data range
                distances_outside = np.minimum(
                    np.abs(X.flatten() - data_min),
                    np.abs(X.flatten() - data_max)
                )
                damping_factor = np.exp(-distances_outside / (0.5 * data_span))
                predictions[outside_mask] *= damping_factor[outside_mask]

        return predictions

    def plot_density_estimate(self, x_range=None, n_points=1000, true_pdf=None,
                              figsize=(12, 6), title="Improved Density Estimation"):
        if x_range is None and self.data_range is not None:
            data_span = self.data_range[1] - self.data_range[0]
            x_range = (self.data_range[0] - 0.2 * data_span,
                       self.data_range[1] + 0.2 * data_span)
        elif x_range is None:
            x_min, x_max = self.training_data.min(), self.training_data.max()
            x_range = (x_min - 0.5, x_max + 0.5)

        x_eval = np.linspace(x_range[0], x_range[1], n_points)
        density_estimates = self.predict(x_eval)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(x_eval, density_estimates, 'b-', linewidth=2,
                     label=f'Improved kn-NN (k1={self.k1})')

        if true_pdf is not None:
            true_density = true_pdf(x_eval)
            axes[0].plot(x_eval, true_density, 'r--', linewidth=2,
                         label='True PDF')

        axes[0].scatter(self.training_data.flatten(),
                        np.zeros_like(self.training_data.flatten()),
                        alpha=0.6, s=20, c='orange', label='Training Data')

        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Density')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        integral = np.trapz(density_estimates, x_eval)
        axes[0].text(0.05, 0.95, f'∫PDF ≈ {integral:.3f}',
                     transform=axes[0].transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        if self.training_history['loss']:
            axes[1].plot(self.training_history['loss'], label='Training Loss')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Progress')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return integral

