import numpy as np
import matplotlib.pyplot as plt
from knn_neural_network import KnNNeuralNetwork
from evaluation_utils import HyperparameterTuning, BaselineComparison


def load_professor_data(data_string):
    # Loading of the dataset given by the professor
    data_points = data_string.strip().split('\n')
    data = np.array([float(x) for x in data_points])
    print(f"Loaded {len(data)} samples from professor's dataset")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Mean: {data.mean():.3f}, Std: {data.std():.3f}")
    return data


def analyze_professor_data(data):
    # Running comprehensive analysis on this new dataset
    print("\n" + "=" * 60)
    print("Analyzing Professor's Dataset")
    print("=" * 60)

    # Defining the true PDF domain (uniform over [0,10] according to professor)
    def true_pdf(x):
        return np.where((x >= 0) & (x <= 10), 0.1, 0)

    # First, we do hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    tuner = HyperparameterTuning(data)
    tuning_results = tuner.comprehensive_tuning(verbose=True)

    # Training final model with optimal parameters
    print("\nTraining final model with optimal parameters (smoothed version):")
    final_model = KnNNeuralNetwork(
        k1=0.1,  # Using best k1 from tuning
        architecture=(50, 30, 20),  # Using best architecture from tuning
        max_iter=4000,  # Increased iterations for convergence
        learning_rate=0.0001,
        random_state=42,
        alpha=0.01
    )
    final_model.fit(data, biased=False, validation_split=0.2, verbose=True)

    # Only plot curves if history exists
    if hasattr(final_model, 'training_history') and final_model.training_history['loss']:
        final_model.plot_training_curves()

    # Plot with explicit range
    final_model.plot_density_estimate(
        x_range=(0, 10),  # Hard-coded to professor's specified domain
        true_pdf=true_pdf,
        title=f"Smoothed kn-NN Density Estimation (k1={final_model.k1}, α={final_model.alpha})"
    )


    print("\nComparing with baseline methods...")
    baseline = BaselineComparison(data)
    baseline.compare_all_methods(final_model, x_range=(0, 10))

    return final_model, tuning_results


def test_improved_model():
    # Testing the improved model on the professor's dataset

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

    data = load_professor_data(data_string)
    model, results = analyze_professor_data(data)

    import pickle
    with open('professor_dataset_results.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'tuning_results': results,
            'data': data
        }, f)
    print("\nAnalysis complete! Results saved to professor_dataset_results.pkl")


def main():
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

    data = load_professor_data(data_string)
    model, results = analyze_professor_data(data)

    import pickle
    with open('professor_dataset_results.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'tuning_results': results,
            'data': data
        }, f)
    print("\nAnalysis complete! Results saved to professor_dataset_results.pkl")


if __name__ == "__main__":
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 10

    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()