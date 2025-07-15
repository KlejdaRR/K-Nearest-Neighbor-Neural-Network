import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class KnNNeuralNetwork:
    def __init__(self, k1=1.0, architecture=(50, 30), max_iter=2000,
                 learning_rate=0.0001, random_state=42, alpha=0.01):
        self.training_history = {
            'loss': [],
            'validation_loss': []
        }
        # Increased alpha
        self.k1 = k1
        self.architecture = architecture
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.alpha = alpha  # Stronger L2 regularization

        self.mlp = None
        self.scaler = StandardScaler()
        self.training_data = None
        self.targets = None
        self.kn = None
        self.data_range = None
        self.target_scale = 1.0

    def _calculate_kn(self, n):
        kn = max(1, int(self.k1 * np.sqrt(n)))
        kn = min(kn, n // 3)  # Cap at reasonable fraction
        return kn

    def _calculate_ball_volume_1d(self, radius):
        return max(2 * radius, 1e-8)

    def _generate_knn_targets(self, X, biased=False):
        n = len(X)
        self.kn = self._calculate_kn(n)
        targets = np.zeros(n)
        self.data_range = (X.min(), X.max())
        data_span = self.data_range[1] - self.data_range[0]

        for i in range(n):
            if biased:
                X_search = X
                n_effective = n
            else:
                X_search = np.delete(X, i, axis=0)
                n_effective = n - 1

            kn_effective = min(self.kn, len(X_search))

            if len(X_search) > 0:
                nbrs = NearestNeighbors(n_neighbors=kn_effective, metric='euclidean')
                nbrs.fit(X_search)
                distances, _ = nbrs.kneighbors(X[i].reshape(1, -1))
                radius = distances[0, -1]

                # Smoother radius handling
                radius = max(radius, data_span / (20 * n))  # Larger minimum radius
                volume = self._calculate_ball_volume_1d(radius)
                density_estimate = kn_effective / (n_effective * volume)

                # Clip extreme values
                max_reasonable_density = 5 / data_span  # Reduced from 10
                targets[i] = np.clip(density_estimate, 0, max_reasonable_density)
            else:
                targets[i] = 0

        return targets

    def fit(self, X, biased=False, validation_split=0.2, verbose=True):
        X = np.array(X).reshape(-1, 1)
        self.training_data = X.copy()
        targets = self._generate_knn_targets(X, biased=biased)
        self.targets = targets

        # Normalize targets more robustly
        target_scale = np.std(targets) + 1e-8
        targets = targets / target_scale
        self.target_scale = target_scale

        X_scaled = self.scaler.fit_transform(X)

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, targets, test_size=validation_split, random_state=self.random_state
            )
        else:
            X_train, y_train = X_scaled, targets
            X_val, y_val = None, None

        self.mlp = MLPRegressor(
            hidden_layer_sizes=self.architecture,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate,
            random_state=self.random_state,
            activation='relu',
            solver='adam',
            alpha=self.alpha,
            early_stopping=True,
            n_iter_no_change=100,
            tol=1e-6
        )
        self.mlp.fit(X_train, y_train)
        # Store training history
        if hasattr(self.mlp, 'loss_curve_'):
            self.training_history['loss'] = self.mlp.loss_curve_
        if hasattr(self.mlp, 'validation_scores_'):
            self.training_history['validation_loss'] = self.mlp.validation_scores_

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        predictions = self.mlp.predict(X_scaled) * self.target_scale
        predictions = np.maximum(predictions, 0)

        # Smoother boundary handling
        if self.data_range is not None:
            data_min, data_max = self.data_range
            data_span = data_max - data_min
            boundary_buffer = 0.2 * data_span  # Increased buffer

            # Apply smoother exponential decay outside range
            left_mask = X.flatten() < data_min
            right_mask = X.flatten() > data_max

            if np.any(left_mask):
                distances = (data_min - X.flatten()[left_mask]) / data_span
                damping = np.exp(-5 * distances)  # Smoother decay
                predictions[left_mask] *= damping

            if np.any(right_mask):
                distances = (X.flatten()[right_mask] - data_max) / data_span
                damping = np.exp(-5 * distances)  # Smoother decay
                predictions[right_mask] *= damping

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

        integral = np.trapezoid(density_estimates, x_eval)
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

    def compare_biased_unbiased(self, X, x_range=None, n_points=1000,
                                true_pdf=None, figsize=(12, 5)):
        # Comparing biased vs unbiased versions side by side
        # Training of biased version
        model_biased = KnNNeuralNetwork(
            k1=self.k1, architecture=self.architecture,
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            random_state=self.random_state, alpha=self.alpha
        )
        model_biased.fit(X, biased=True, verbose=False)

        # Training of unbiased version
        model_unbiased = KnNNeuralNetwork(
            k1=self.k1, architecture=self.architecture,
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            random_state=self.random_state, alpha=self.alpha
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

