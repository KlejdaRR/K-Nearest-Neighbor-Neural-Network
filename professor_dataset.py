import numpy as np
import matplotlib.pyplot as plt
from knn_neural_network import KnNNeuralNetwork


def test_improved_model():
    """Test the improved model on the professor's dataset"""

    # Professor's data
    data_string = """9.338509e+00
8.507600e+00
9.948655e+00
8.173691e+00
9.455406e+00
8.992554e+00
9.421297e+00
9.799669e+00
8.513956e+00
9.968746e+00
9.651385e+00
9.404957e+00
8.570448e+00
8.550490e+00
9.270396e+00
8.122849e+00
8.851571e+00
8.202571e+00
9.139815e+00
9.987814e+00
9.175507e+00
9.784623e+00
5.888731e+00
9.815300e+00
9.749408e+00
7.975595e+00
7.743580e+00
8.544786e+00
8.717772e+00
6.285266e+00
9.572749e+00
9.652127e+00
9.680381e+00
9.298379e+00
7.810104e+00
9.754563e+00
9.974293e+00
7.609780e+00
9.024825e+00
9.698005e+00
9.524585e+00
8.015665e+00
7.072561e+00
9.380539e+00
8.802788e+00
8.636000e+00
9.372733e+00
8.702384e+00
9.137634e+00
8.865982e+00
7.811190e+00
8.276196e+00
9.957435e+00
9.756414e+00
7.855027e+00
9.989560e+00
9.608489e+00
5.833307e+00
9.785972e+00
8.425355e+00
9.871499e+00
7.923205e+00
9.313694e+00
9.414033e+00
9.402754e+00
9.068302e+00
8.379124e+00
9.637105e+00
9.279458e+00
8.656078e+00
9.527650e+00
9.961235e+00
9.774344e+00
9.371791e+00
9.343755e+00
9.657895e+00
9.862570e+00
8.570589e+00
9.435012e+00
9.375019e+00
9.587764e+00
8.774345e+00
8.650779e+00
9.149315e+00
9.917333e+00
9.898566e+00
9.544954e+00
9.578945e+00
9.668651e+00
9.042328e+00
9.607844e+00
9.932990e+00
9.016578e+00
9.777700e+00
9.784464e+00
8.116712e+00
9.571859e+00
9.523060e+00
9.700784e+00
9.680427e+00"""

    data = np.array([float(x) for x in data_string.strip().split('\n')])
    print(f"Loaded {len(data)} samples")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Mean: {data.mean():.3f}, Std: {data.std():.3f}")

    # True PDF (uniform over [0,10])
    def true_pdf(x):
        return np.where((x >= 0) & (x <= 10), 0.1, 0)

    # Test different k1 values
    k1_values = [0.5, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, k1 in enumerate(k1_values):
        print(f"\nTesting k1 = {k1}")

        model = KnNNeuralNetwork(
            k1=k1,
            architecture=(100, 50, 25),  # Deeper network
            max_iter=3000,
            learning_rate=0.0001,
            alpha=0.01,  # More regularization
            random_state=42
        )

        model.fit(data, biased=False, validation_split=0.2, verbose=True)

        # Evaluate on extended range
        x_eval = np.linspace(0, 10, 1000)
        density_est = model.predict(x_eval)
        true_density = true_pdf(x_eval)

        # Plot
        axes[i].plot(x_eval, density_est, 'b-', linewidth=2,
                     label=f'Improved kn-NN (k1={k1})')
        axes[i].plot(x_eval, true_density, 'r--', linewidth=2,
                     label='True PDF (Uniform)')
        axes[i].scatter(data, np.zeros_like(data), alpha=0.6, s=15,
                        c='orange', label='Data')

        # Calculate integral
        integral = np.trapz(density_est, x_eval)
        axes[i].set_title(f'k1={k1}, ∫PDF={integral:.3f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 0.5)  # Reasonable y-axis limit

        print(f"Integral: {integral:.3f}")
        print(f"Max density: {density_est.max():.3f}")

    plt.tight_layout()
    plt.show()


# Run the test
if __name__ == "__main__":
    test_improved_model()