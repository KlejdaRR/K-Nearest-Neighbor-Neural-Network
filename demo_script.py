import matplotlib.pyplot as plt
import argparse
import os

from knn_neural_network import KnNNeuralNetwork, generate_test_data, run_experiments
from evaluation_utils import (
    ModelEvaluator, BaselineComparison, HyperparameterTuning,
    ExperimentRunner, evaluate_on_instructor_dataset
)


def basic_demonstration():
    # Basic demo to show how the kn-NN neural network works
    # We use this to quickly test if everything is working properly


    print("=" * 60)
    print("Basic kn-NN Neural Network Demonstration")
    print("=" * 60)

    print("1. Generating synthetic data from mixture distribution...")
    data, true_pdf = generate_test_data('mixture', n_samples=200, random_state=42)

    print(f"   Generated {len(data)} samples")
    print(f"   Data range: [{data.min():.2f}, {data.max():.2f}]")

    print("\n2. Training kn-NN Neural Network...")
    model = KnNNeuralNetwork(
        k1=1.0,  # This seems to work well for most cases
        architecture=(50, 30),  # Two hidden layers - good balance
        max_iter=500,  # Usually converges before this
        learning_rate=0.001,  # Standard learning rate
        random_state=42  # For reproducibility
    )

    model.fit(data, biased=False, validation_split=0.2, verbose=True)

    print("\n3. Visualizing results...")
    model.plot_training_curves()

    model.plot_density_estimate(
        true_pdf=true_pdf,
        title="kn-NN Neural Network Density Estimation"
    )

    print("\n4. Comparing biased vs unbiased versions...")
    model.compare_biased_unbiased(data, true_pdf=true_pdf)

    print("\n5. Comparing with baseline methods...")
    baseline = BaselineComparison(data)
    baseline.compare_all_methods(model, true_pdf=true_pdf)

    print("\n6. Basic demonstration completed!")

    return model, data, true_pdf


def hyperparameter_tuning_demo():
    # Showing how to do automatic hyperparameter tuning


    print("=" * 60)
    print("Hyperparameter Tuning Demonstration")
    print("=" * 60)

    data, true_pdf = generate_test_data('mixture', n_samples=300, random_state=42)

    tuner = HyperparameterTuning(data, true_pdf=true_pdf)

    print("Starting comprehensive hyperparameter tuning...")
    print("This might take a few minutes...")
    tuning_results = tuner.comprehensive_tuning(verbose=True)

    print(f"\nTraining final model with optimal parameters:")
    print(f"  k1: {tuning_results['best_k1']}")
    print(f"  architecture: {tuning_results['best_architecture']}")

    # Training the final optimized model
    final_model = KnNNeuralNetwork(
        k1=tuning_results['best_k1'],
        architecture=tuning_results['best_architecture'],
        max_iter=1000,  # Using more iterations for final model
        learning_rate=0.001,
        random_state=42
    )

    final_model.fit(data, biased=False, validation_split=0.2, verbose=True)

    # Showing the optimized result
    final_model.plot_density_estimate(
        true_pdf=true_pdf,
        title=f"Optimized kn-NN (k1={tuning_results['best_k1']}, arch={tuning_results['best_architecture']})"
    )

    return final_model, tuning_results


def sample_size_experiment():
    # Experimenting to see how performance changes with dataset size


    print("=" * 60)
    print("Sample Size Experiment")
    print("=" * 60)

    # Generating a large dataset to subsample from
    data, true_pdf = generate_test_data('mixture', n_samples=1000, random_state=42)

    experiment_runner = ExperimentRunner(data, true_pdf=true_pdf)
    results = experiment_runner.run_sample_size_experiment(
        sample_sizes=[50, 100, 200, 400, 800],
        k1=1.0,
        architecture=(50, 30),
        n_runs=3
    )

    sample_sizes = list(results.keys())
    ise_means = [results[n]['ise_mean'] for n in sample_sizes]
    ise_stds = [results[n]['ise_std'] for n in sample_sizes]

    plt.figure(figsize=(10, 6))
    plt.errorbar(sample_sizes, ise_means, yerr=ise_stds,
                 marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Sample Size')
    plt.ylabel('Integrated Squared Error')
    plt.title('Sample Size vs. ISE Performance')
    plt.grid(True, alpha=0.3)
    plt.show()

    return results


def instructor_dataset_demo(data_path):
    # Evaluating on the instructor's dataset


    print("=" * 60)
    print("Instructor Dataset Evaluation")
    print("=" * 60)

    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found!")
        print("Please provide a valid path to the instructor's dataset.")
        return None

    try:
        model, tuning_results = evaluate_on_instructor_dataset(data_path)
        print("\nInstructor dataset evaluation completed successfully!")
        return model, tuning_results
    except Exception as e:
        print(f"Error evaluating instructor dataset: {e}")
        return None


def full_experiments():
    # Running all experiments

    print("=" * 80)
    print("COMPREHENSIVE kn-NN NEURAL NETWORK EXPERIMENTS")
    print("=" * 80)

    print("\n" + "=" * 40)
    print("EXPERIMENT 1: Basic Demonstration")
    print("=" * 40)
    basic_model, basic_data, basic_true_pdf = basic_demonstration()

    print("\n" + "=" * 40)
    print("EXPERIMENT 2: Hyperparameter Tuning")
    print("=" * 40)
    tuned_model, tuning_results = hyperparameter_tuning_demo()

    print("\n" + "=" * 40)
    print("EXPERIMENT 3: Sample Size Analysis")
    print("=" * 40)
    sample_results = sample_size_experiment()

    print("\n" + "=" * 40)
    print("EXPERIMENT 4: Different Distributions")
    print("=" * 40)

    # Testing on various distributions to see generalization
    distributions = ['normal', 'exponential', 'uniform', 'mixture']

    for dist in distributions:
        print(f"\nTesting on {dist} distribution...")
        data, true_pdf = generate_test_data(dist, n_samples=300, random_state=42)

        model = KnNNeuralNetwork(k1=1.0, architecture=(50, 30), max_iter=500)
        model.fit(data, biased=False, verbose=False)

        model.plot_density_estimate(
            true_pdf=true_pdf,
            title=f"kn-NN on {dist.capitalize()} Distribution"
        )

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)

    return {
        'basic_model': basic_model,
        'tuned_model': tuned_model,
        'tuning_results': tuning_results,
        'sample_results': sample_results
    }


def main():

    parser = argparse.ArgumentParser(
        description='kn-NN Neural Network Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_script.py                           # Basic demo
  python demo_script.py --full_experiments        # Run all experiments
  python demo_script.py --instructor_data data.txt # Evaluating on instructor's data
  python demo_script.py --hyperparameter_tuning   # Just hyperparameter tuning
  python demo_script.py --distribution normal --k1 1.5  # Custom parameters
        """
    )

    parser.add_argument(
        '--instructor_data',
        type=str,
        help='Path to instructor\'s dataset file'
    )

    parser.add_argument(
        '--full_experiments',
        action='store_true',
        help='Run all comprehensive experiments'
    )

    parser.add_argument(
        '--hyperparameter_tuning',
        action='store_true',
        help='Run hyperparameter tuning demonstration'
    )

    parser.add_argument(
        '--sample_size_experiment',
        action='store_true',
        help='Run sample size experiment'
    )

    parser.add_argument(
        '--distribution',
        choices=['normal', 'exponential', 'uniform', 'mixture'],
        default='mixture',
        help='Distribution to use for basic demonstration (default: mixture)'
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=200,
        help='Number of samples to generate (default: 200)'
    )

    parser.add_argument(
        '--k1',
        type=float,
        default=1.0,
        help='k1 parameter for kn-NN (default: 1.0)'
    )

    parser.add_argument(
        '--architecture',
        type=str,
        default='50,30',
        help='Neural network architecture as comma-separated integers (default: 50,30)'
    )

    args = parser.parse_args()

    try:
        architecture = tuple(map(int, args.architecture.split(',')))
    except:
        print("Error: Invalid architecture format. Use comma-separated integers like '50,30'")
        return

    # Route to appropriate function based on arguments
    if args.instructor_data:
        instructor_dataset_demo(args.instructor_data)

    elif args.full_experiments:
        full_experiments()

    elif args.hyperparameter_tuning:
        hyperparameter_tuning_demo()

    elif args.sample_size_experiment:
        sample_size_experiment()

    else:
        # Default: basic demonstration with custom parameters
        print("=" * 60)
        print(f"Basic Demonstration - {args.distribution.capitalize()} Distribution")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  Distribution: {args.distribution}")
        print(f"  Number of samples: {args.n_samples}")
        print(f"  k1 parameter: {args.k1}")
        print(f"  Architecture: {architecture}")
        print()

        # Generating test data
        data, true_pdf = generate_test_data(
            args.distribution,
            n_samples=args.n_samples,
            random_state=42
        )

        model = KnNNeuralNetwork(
            k1=args.k1,
            architecture=architecture,
            max_iter=500,
            learning_rate=0.001,
            random_state=42
        )

        # Training the model
        model.fit(data, biased=False, validation_split=0.2, verbose=True)

        model.plot_training_curves()
        model.plot_density_estimate(
            true_pdf=true_pdf,
            title=f"kn-NN on {args.distribution.capitalize()} Distribution"
        )

        baseline = BaselineComparison(data)
        baseline.compare_all_methods(model, true_pdf=true_pdf)


if __name__ == "__main__":
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 10
    plt.rcParams['lines.linewidth'] = 2

    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()