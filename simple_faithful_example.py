import numpy as np
import matplotlib.pyplot as plt
from knn_neural_network import KnNNeuralNetwork


def load_faithful_data(filename='datasets/faithful.dat.txt', variable='eruptions'):    # Loading the Old Faithful dataset
    # Parameters:
    # filename: path to the data file
    # variable: 'eruptions' for eruption duration or 'waiting' for waiting time

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Finding where the actual data starts (skipping header)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit():
                data_start = i
                break

        eruptions = []
        waiting = []

        # Parsing the data
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    eruptions.append(float(parts[1]))  # Eruption duration
                    waiting.append(float(parts[2]))  # Waiting time
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
        print("Please make sure faithful.dat.txt is in the current directory")
        return None


def run_faithful_analysis():
    #Running a complete analysis of the Old Faithful dataset

    print("=" * 60)
    print("Old Faithful Geyser Analysis with kn-NN Neural Network")
    print("=" * 60)

    # Loading eruption duration data
    eruption_data = load_faithful_data(variable='eruptions')
    if eruption_data is None:
        return

    print(f"\nDataset Statistics:")
    print(f"  Mean eruption duration: {eruption_data.mean():.2f} minutes")
    print(f"  Standard deviation: {eruption_data.std():.2f} minutes")
    print(f"  Sample size: {len(eruption_data)} observations")

    # Training the kn-NN model
    print(f"\nTraining kn-NN Neural Network...")
    model = KnNNeuralNetwork(
        k1=1.0,  # Number of neighbors parameter
        architecture=(100, 50),  # Two hidden layers
        max_iter=2000,  # Maximum training iterations
        learning_rate=0.0001,
        random_state=42
    )

    # Fitting the model (unbiased version)
    model.fit(eruption_data, biased=False, validation_split=0.2, verbose=True)

    # Creating visualizations
    print(f"\nGenerating visualizations...")

    # Plot 1: Training curves
    model.plot_training_curves()

    # Plot 2: Density estimate
    model.plot_density_estimate(
        title="Old Faithful Eruption Duration - Density Estimation"
    )

    # Plot 3: Comparing biased vs unbiased
    model.compare_biased_unbiased(eruption_data)

    # Plot 4: Data distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Histogram of raw data
    axes[0, 0].hist(eruption_data, bins=20, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Eruption Duration Histogram')
    axes[0, 0].set_xlabel('Duration (minutes)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)

    # kn-NN density estimate
    x_range = (eruption_data.min() - 0.5, eruption_data.max() + 0.5)
    x_eval = np.linspace(x_range[0], x_range[1], 1000)
    density_est = model.predict(x_eval)

    axes[0, 1].plot(x_eval, density_est, 'b-', linewidth=2, label='kn-NN Estimate')
    axes[0, 1].scatter(eruption_data, np.zeros_like(eruption_data),
                       alpha=0.6, s=20, c='orange', label='Data Points')
    axes[0, 1].set_title('kn-NN Density Estimate')
    axes[0, 1].set_xlabel('Duration (minutes)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Box plot
    axes[1, 0].boxplot(eruption_data, vert=True)
    axes[1, 0].set_title('Eruption Duration Box Plot')
    axes[1, 0].set_ylabel('Duration (minutes)')
    axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics
    stats_text = f"""Summary Statistics:

Sample Size: {len(eruption_data)}
Mean: {eruption_data.mean():.3f} min
Median: {np.median(eruption_data):.3f} min
Std Dev: {eruption_data.std():.3f} min
Min: {eruption_data.min():.3f} min
Max: {eruption_data.max():.3f} min

Model Info:
k1 parameter: {model.k1}
kn (neighbors used): {model.kn}
Architecture: {model.architecture}
Training iterations: {model.mlp.n_iter_}
Final loss: {model.mlp.loss_:.6f}"""

    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Dataset & Model Summary')

    plt.tight_layout()
    plt.show()

    # Simple comparison with KDE
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV

    print(f"\nComparing with Kernel Density Estimation...")

    # Find optimal bandwidth for KDE
    bandwidths = np.logspace(-1, 1, 20)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths}, cv=5)
    grid.fit(eruption_data.reshape(-1, 1))

    kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_params_['bandwidth'])
    kde.fit(eruption_data.reshape(-1, 1))
    kde_density = np.exp(kde.score_samples(x_eval.reshape(-1, 1)))

    # Comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_eval, density_est, 'b-', linewidth=2, label='kn-NN Neural Network')
    plt.plot(x_eval, kde_density, 'r--', linewidth=2, label=f'KDE (bandwidth={grid.best_params_["bandwidth"]:.3f})')
    plt.scatter(eruption_data, np.zeros_like(eruption_data),
                alpha=0.6, s=15, c='orange', label='Data Points')
    plt.title('Density Estimation Comparison: kn-NN vs KDE')
    plt.xlabel('Eruption Duration (minutes)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\nAnalysis complete!")
    print(f"The kn-NN method has been successfully applied to real geyser data.")

    return model, eruption_data


def analyze_waiting_times():
    # Analyze waiting times instead of eruption duration

    print("=" * 60)
    print("Old Faithful Waiting Times Analysis")
    print("=" * 60)

    waiting_data = load_faithful_data(variable='waiting')
    if waiting_data is None:
        return

    print(f"\nTraining model on waiting times...")
    model = KnNNeuralNetwork(k1=1.0, architecture=(100, 50), max_iter=2000)
    model.fit(waiting_data, biased=False, validation_split=0.2, verbose=True)

    model.plot_density_estimate(
        title="Old Faithful Waiting Times - Density Estimation"
    )

    return model, waiting_data


if __name__ == "__main__":
    # Setting up plotting
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 10
    plt.rcParams['lines.linewidth'] = 2

    try:
        # Running the main analysis
        eruption_model, eruption_data = run_faithful_analysis()

        # Optionally analyze waiting times too
        print("\n" + "=" * 60)
        print("Would you like to analyze waiting times as well? (y/n)")
        response = input().lower().strip()

        if response == 'y' or response == 'yes':
            waiting_model, waiting_data = analyze_waiting_times()

            # Comparing both variables
            print("\n" + "=" * 60)
            print("Comparing Eruption Duration vs Waiting Times")
            print("=" * 60)

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Eruption duration
            x_range_e = (eruption_data.min() - 0.5, eruption_data.max() + 0.5)
            x_eval_e = np.linspace(x_range_e[0], x_range_e[1], 1000)
            density_e = eruption_model.predict(x_eval_e)

            axes[0].plot(x_eval_e, density_e, 'b-', linewidth=2, label='kn-NN Estimate')
            axes[0].scatter(eruption_data, np.zeros_like(eruption_data),
                            alpha=0.6, s=15, c='orange', label='Data Points')
            axes[0].set_title('Eruption Duration')
            axes[0].set_xlabel('Duration (minutes)')
            axes[0].set_ylabel('Density')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Waiting times
            x_range_w = (waiting_data.min() - 5, waiting_data.max() + 5)
            x_eval_w = np.linspace(x_range_w[0], x_range_w[1], 1000)
            density_w = waiting_model.predict(x_eval_w)

            axes[1].plot(x_eval_w, density_w, 'g-', linewidth=2, label='kn-NN Estimate')
            axes[1].scatter(waiting_data, np.zeros_like(waiting_data),
                            alpha=0.6, s=15, c='orange', label='Data Points')
            axes[1].set_title('Waiting Times')
            axes[1].set_xlabel('Time (minutes)')
            axes[1].set_ylabel('Density')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Showing correlation
            correlation = np.corrcoef(eruption_data, waiting_data)[0, 1]
            print(f"\nCorrelation between eruption duration and waiting time: {correlation:.3f}")

            plt.figure(figsize=(10, 6))
            plt.scatter(eruption_data, waiting_data, alpha=0.6, s=30)
            plt.xlabel('Eruption Duration (minutes)')
            plt.ylabel('Waiting Time (minutes)')
            plt.title(f'Eruption Duration vs Waiting Time (correlation = {correlation:.3f})')
            plt.grid(True, alpha=0.3)
            plt.show()

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()