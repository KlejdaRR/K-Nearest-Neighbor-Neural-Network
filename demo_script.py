import matplotlib.pyplot as plt
import argparse
import numpy as np
from knn_neural_network import KnNNeuralNetwork
from evaluation_utils import (BaselineComparison, HyperparameterTuning,
                              ExperimentRunner)


def load_faithful_data(filename='faithful.dat.txt', variable='eruptions'):
     # Basic demo to show how the kn-NN neural network works
    # We use this to quickly test if everything is working properly
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Finding where the actual data starts
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit():
                data_start = i
                break

        eruptions = []
        waiting = []

        # Parse the data
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    eruptions.append(float(parts[1]))
                    waiting.append(float(parts[2]))
                except ValueError:
                    continue

        if variable == 'eruptions':
            data = np.array(eruptions)
            print(f"Loaded {len(data)} eruption duration measurements")
            print(f"Range: {data.min():.3f} to {data.max():.3f} minutes")
        elif variable == 'waiting':
            data = np.array(waiting)
            print(f"Loaded {len(data)} waiting time measurements")
            print(f"Range: {data.min():.0f} to {data.max():.0f} minutes")
        else:
            raise ValueError("Variable must be 'eruptions' or 'waiting'")

        return data

    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        print("Please make sure the faithful.dat.txt file is in the current directory")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def basic_demonstration(variable='eruptions'):
    # Basic demo that uses Old Faithful data

    print("=" * 60)
    print(f"Basic kn-NN Neural Network Demonstration")
    print(f"Dataset: Old Faithful Geyser - {variable.capitalize()}")
    print("=" * 60)

    print(f"1. Loading Old Faithful {variable} data...")
    data = load_faithful_data(variable=variable)
    if data is None:
        return None, None

    print(f"   Loaded {len(data)} samples")
    print(f"   Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"   Mean: {data.mean():.3f}, Std: {data.std():.3f}")

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
        title=f"kn-NN Density Estimation - Old Faithful {variable.capitalize()}"
    )

    print("\n4. Comparing biased vs unbiased versions...")
    model.compare_biased_unbiased(data)

    print("\n5. Comparing with baseline methods...")
    baseline = BaselineComparison(data)
    baseline.compare_all_methods(model)

    print(f"\n6. Basic demonstration completed for {variable}!")
    return model, data


def hyperparameter_tuning_demo(variable='eruptions'):
    # Showing how to do automatic hyperparameter tuning
    print("=" * 60)
    print("Hyperparameter Tuning Demonstration")
    print(f"Dataset: Old Faithful Geyser - {variable.capitalize()}")
    print("=" * 60)

    data = load_faithful_data(variable=variable)
    if data is None:
        return None, None

    tuner = HyperparameterTuning(data)
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
        title=f"Optimized kn-NN - Old Faithful {variable.capitalize()} (k1={tuning_results['best_k1']}, arch={tuning_results['best_architecture']})"
    )

    return final_model, tuning_results


def sample_size_experiment(variable='eruptions'):
    # Experimenting to see how performance changes with dataset size
    print("=" * 60)
    print("Sample Size Experiment")
    print(f"Dataset: Old Faithful Geyser - {variable.capitalize()}")
    print("=" * 60)
    # Generating a large dataset to subsample from

    data = load_faithful_data(variable=variable)
    if data is None:
        return None

    experiment_runner = ExperimentRunner(data)

    # Adjusting sample sizes based on the actual data size (272 observations)
    max_size = len(data)
    sample_sizes = [30, 50, 100, 150, 200, int(max_size * 0.9)]
    sample_sizes = [s for s in sample_sizes if s < max_size]

    results = experiment_runner.run_sample_size_experiment(
        sample_sizes=sample_sizes,
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
    plt.ylabel('Training Loss (Proxy for ISE)')
    plt.title(f'Sample Size vs. Performance - Old Faithful {variable.capitalize()}')
    plt.grid(True, alpha=0.3)
    plt.show()

    return results


def compare_variables():
    # Comparing density estimation for both eruptions and waiting times
    print("=" * 60)
    print("Comparing Eruptions vs Waiting Times")
    print("=" * 60)

    # Loading both variables
    eruptions_data = load_faithful_data(variable='eruptions')
    waiting_data = load_faithful_data(variable='waiting')

    if eruptions_data is None or waiting_data is None:
        return None

    # Training models for both variables
    models = {}
    for var_name, data in [('eruptions', eruptions_data), ('waiting', waiting_data)]:
        print(f"\nTraining model for {var_name}...")
        model = KnNNeuralNetwork(k1=1.0, architecture=(50, 30), max_iter=500)
        model.fit(data, biased=False, verbose=False)
        models[var_name] = {'model': model, 'data': data}

    # Creating comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, (var_name, info) in enumerate(models.items()):
        model = info['model']
        data = info['data']

        x_min, x_max = data.min(), data.max()
        x_range = (x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1)
        x_eval = np.linspace(x_range[0], x_range[1], 1000)
        density_est = model.predict(x_eval)

        axes[i].plot(x_eval, density_est, 'b-', linewidth=2, label='kn-NN Estimate')
        axes[i].scatter(data, np.zeros_like(data), alpha=0.6, s=20, c='orange', label='Data Points')

        axes[i].set_title(f'Old Faithful {var_name.capitalize()}')
        axes[i].set_xlabel(f'{var_name.capitalize()} {"(minutes)" if var_name == "eruptions" else "(minutes)"}')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return models


def faithful_dataset_analysis():
    # Comprehensive analysis of the Old Faithful dataset
    print("=" * 80)
    print("COMPREHENSIVE OLD FAITHFUL DATASET ANALYSIS")
    print("=" * 80)

    # Loading both variables
    eruptions_data = load_faithful_data(variable='eruptions')
    waiting_data = load_faithful_data(variable='waiting')

    if eruptions_data is None or waiting_data is None:
        return None

    print("\n" + "=" * 40)
    print("ANALYSIS 1: Basic Demonstration - Eruptions")
    print("=" * 40)
    eruptions_model, _ = basic_demonstration('eruptions')

    print("\n" + "=" * 40)
    print("ANALYSIS 2: Basic Demonstration - Waiting Times")
    print("=" * 40)
    waiting_model, _ = basic_demonstration('waiting')

    print("\n" + "=" * 40)
    print("ANALYSIS 3: Hyperparameter Tuning - Eruptions")
    print("=" * 40)
    tuned_eruptions_model, eruptions_tuning = hyperparameter_tuning_demo('eruptions')

    print("\n" + "=" * 40)
    print("ANALYSIS 4: Sample Size Analysis - Eruptions")
    print("=" * 40)
    eruptions_sample_results = sample_size_experiment('eruptions')

    print("\n" + "=" * 40)
    print("ANALYSIS 5: Variable Comparison")
    print("=" * 40)
    comparison_models = compare_variables()

    print("\n" + "=" * 40)
    print("ANALYSIS 6: Data Distribution Analysis")
    print("=" * 40)

    # Creating comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Histograms
    axes[0, 0].hist(eruptions_data, bins=20, alpha=0.7, density=True, label='Eruptions')
    axes[0, 0].set_title('Eruption Duration Distribution')
    axes[0, 0].set_xlabel('Duration (minutes)')
    axes[0, 0].set_ylabel('Density')

    axes[0, 1].hist(waiting_data, bins=20, alpha=0.7, density=True, label='Waiting Time', color='orange')
    axes[0, 1].set_title('Waiting Time Distribution')
    axes[0, 1].set_xlabel('Time (minutes)')
    axes[0, 1].set_ylabel('Density')

    # Scattering plot of relationship
    axes[0, 2].scatter(eruptions_data, waiting_data, alpha=0.6)
    axes[0, 2].set_title('Eruptions vs Waiting Time')
    axes[0, 2].set_xlabel('Eruption Duration (minutes)')
    axes[0, 2].set_ylabel('Waiting Time (minutes)')

    # kn-NN density estimates
    if eruptions_model:
        x_eval_e = np.linspace(eruptions_data.min() - 0.5, eruptions_data.max() + 0.5, 1000)
        density_e = eruptions_model.predict(x_eval_e)
        axes[1, 0].plot(x_eval_e, density_e, 'b-', linewidth=2, label='kn-NN')
        axes[1, 0].scatter(eruptions_data, np.zeros_like(eruptions_data), alpha=0.4, s=10, c='orange')
        axes[1, 0].set_title('kn-NN: Eruption Duration')
        axes[1, 0].set_xlabel('Duration (minutes)')
        axes[1, 0].set_ylabel('Density')

    if waiting_model:
        x_eval_w = np.linspace(waiting_data.min() - 5, waiting_data.max() + 5, 1000)
        density_w = waiting_model.predict(x_eval_w)
        axes[1, 1].plot(x_eval_w, density_w, 'g-', linewidth=2, label='kn-NN')
        axes[1, 1].scatter(waiting_data, np.zeros_like(waiting_data), alpha=0.4, s=10, c='orange')
        axes[1, 1].set_title('kn-NN: Waiting Time')
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Density')

    # Summary statistics
    stats_text = f"""Old Faithful Dataset Summary:

Eruption Duration:
  Mean: {eruptions_data.mean():.2f} min
  Std:  {eruptions_data.std():.2f} min
  Range: {eruptions_data.min():.2f} - {eruptions_data.max():.2f} min

Waiting Time:
  Mean: {waiting_data.mean():.1f} min  
  Std:  {waiting_data.std():.1f} min
  Range: {waiting_data.min():.0f} - {waiting_data.max():.0f} min

Correlation: {np.corrcoef(eruptions_data, waiting_data)[0, 1]:.3f}
Sample Size: {len(eruptions_data)} observations"""

    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Dataset Statistics')

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print("OLD FAITHFUL ANALYSIS COMPLETED!")
    print("=" * 80)

    return {
        'eruptions_model': eruptions_model,
        'waiting_model': waiting_model,
        'tuned_eruptions_model': tuned_eruptions_model,
        'eruptions_tuning': eruptions_tuning,
        'eruptions_sample_results': eruptions_sample_results,
        'comparison_models': comparison_models
    }


def main():
    parser = argparse.ArgumentParser(
        description='kn-NN Neural Network Demonstration with Old Faithful Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_script_faithful.py                           # Basic demo with eruptions
  python demo_script_faithful.py --variable waiting        # Basic demo with waiting times
  python demo_script_faithful.py --full_analysis           # Running all analyses
  python demo_script_faithful.py --hyperparameter_tuning   # Just hyperparameter tuning
  python demo_script_faithful.py --compare_variables       # Comparing eruptions vs waiting
  python demo_script_faithful.py --variable eruptions --k1 1.5  # Custom parameters
        """
    )

    parser.add_argument(
        '--data_file',
        type=str,
        default='faithful.dat.txt',
        help='Path to the Old Faithful dataset file (default: faithful.dat.txt)'
    )

    parser.add_argument(
        '--variable',
        choices=['eruptions', 'waiting'],
        default='eruptions',
        help='Which variable to analyze (default: eruptions)'
    )

    parser.add_argument(
        '--full_analysis',
        action='store_true',
        help='Run comprehensive analysis of Old Faithful dataset'
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
        '--compare_variables',
        action='store_true',
        help='Compare eruptions vs waiting time analysis'
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
    if args.full_analysis:
        faithful_dataset_analysis()
    elif args.hyperparameter_tuning:
        hyperparameter_tuning_demo(args.variable)
    elif args.sample_size_experiment:
        sample_size_experiment(args.variable)
    elif args.compare_variables:
        compare_variables()
    else:
        # Default: basic demonstration with custom parameters
        print("=" * 60)
        print(f"Old Faithful Dataset Analysis - {args.variable.capitalize()}")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  Variable: {args.variable}")
        print(f"  k1 parameter: {args.k1}")
        print(f"  Architecture: {architecture}")
        print(f"  Data file: {args.data_file}")
        print()

        # Loading the data
        data = load_faithful_data(args.data_file, args.variable)
        if data is None:
            return

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
            title=f"kn-NN Density Estimation - Old Faithful {args.variable.capitalize()}"
        )

        baseline = BaselineComparison(data)
        baseline.compare_all_methods(model)


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