import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

class ModelEvaluator:
    # Helper class for evaluating density estimation models
    # We needed this to compare our results with the true PDF


    @staticmethod
    def integrated_squared_error(estimated_pdf, true_pdf, x_range, n_points=10000):
        # Calculating ISE between estimated and true PDF
        # This is one of the standard metrics for density estimation

        x_eval = np.linspace(x_range[0], x_range[1], n_points)
        dx = (x_range[1] - x_range[0]) / n_points

        estimated_vals = estimated_pdf(x_eval)
        true_vals = true_pdf(x_eval)

        squared_error = (estimated_vals - true_vals) ** 2
        ise = np.trapz(squared_error, dx=dx)

        return ise

    @staticmethod
    def mean_absolute_error(estimated_pdf, true_pdf, x_range, n_points=10000):

        # Calculating MAE - because it is easier to interpret than ISE

        x_eval = np.linspace(x_range[0], x_range[1], n_points)

        estimated_vals = estimated_pdf(x_eval)
        true_vals = true_pdf(x_eval)

        mae = np.mean(np.abs(estimated_vals - true_vals))

        return mae

    @staticmethod
    def kullback_leibler_divergence(estimated_pdf, true_pdf, x_range, n_points=10000):
        # KL divergence - measures how much the estimated PDF differs from true PDF
        # We Had some issues with numerical stability here, added epsilon to avoid log(0)

        x_eval = np.linspace(x_range[0], x_range[1], n_points)
        dx = (x_range[1] - x_range[0]) / n_points

        true_vals = true_pdf(x_eval)
        estimated_vals = estimated_pdf(x_eval)

        epsilon = 1e-10
        estimated_vals = np.maximum(estimated_vals, epsilon)
        true_vals = np.maximum(true_vals, epsilon)

        valid_mask = true_vals > epsilon

        if np.sum(valid_mask) > 0:
            kl_div = np.trapz(
                true_vals[valid_mask] * np.log(true_vals[valid_mask] / estimated_vals[valid_mask]),
                dx=dx
            )
        else:
            kl_div = np.inf

        return kl_div

    @staticmethod
    def check_normalization(pdf_func, x_range, n_points=10000):
        # Checking if the PDF integrates to 1 (as it should)
        # This helped us debug issues with our implementation

        x_eval = np.linspace(x_range[0], x_range[1], n_points)
        pdf_vals = pdf_func(x_eval)
        integral = np.trapz(pdf_vals, x_eval)
        return integral


class BaselineComparison:
    # Comparing our kn-NN method with standard baseline methods
    # With this we can know for sure if our approach actually works


    def __init__(self, data):
        self.data = data.reshape(-1, 1) if data.ndim == 1 else data
        self.fitted_models = {}

    def fit_parzen_window(self, bandwidth=None):
        # Fitting Parzen window (KDE) with automatic bandwidth selection

        if bandwidth is None:
            # Using cross-validation to find best bandwidth
            bandwidths = np.logspace(-2, 1, 20)
            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                {'bandwidth': bandwidths},
                                cv=5)
            grid.fit(self.data)
            bandwidth = grid.best_params_['bandwidth']

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(self.data)

        self.fitted_models['parzen'] = {
            'model': kde,
            'bandwidth': bandwidth,
            'name': f'Parzen Window (bw={bandwidth:.3f})'
        }

        return kde

    def fit_histogram(self, bins='auto'):
        # Simple histogram-based density estimation

        if bins == 'auto':
            # Use sqrt rule for number of bins
            bins = int(np.sqrt(len(self.data)))

        hist, bin_edges = np.histogram(self.data.flatten(), bins=bins, density=True)

        def histogram_pdf(x):
            # Convert histogram to a function we can evaluate anywhere
            x = np.atleast_1d(x)
            result = np.zeros_like(x)

            for i in range(len(bin_edges) - 1):
                mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
                result[mask] = hist[i]

            mask = x == bin_edges[-1]
            result[mask] = hist[-1]

            return result

        self.fitted_models['histogram'] = {
            'model': histogram_pdf,
            'bins': bins,
            'name': f'Histogram ({bins} bins)'
        }

        return histogram_pdf

    def compare_all_methods(self, knn_model, true_pdf=None, x_range=None,
                            n_points=1000, figsize=(15, 10)):
        # Comparing all methods side by side with quantitative metrics

        if x_range is None:
            x_min, x_max = self.data.min(), self.data.max()
            x_range = (x_min - 1, x_max + 1)

        self.fit_parzen_window()
        self.fit_histogram()

        x_eval = np.linspace(x_range[0], x_range[1], n_points)

        # Getting predictions from all methods
        knn_pred = knn_model.predict(x_eval)
        parzen_pred = np.exp(self.fitted_models['parzen']['model'].score_samples(x_eval.reshape(-1, 1)))
        hist_pred = self.fitted_models['histogram']['model'](x_eval)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        methods = [
            ('kn-NN Neural Network', knn_pred, 'blue'),
            ('Parzen Window', parzen_pred, 'green'),
            ('Histogram', hist_pred, 'purple')
        ]

        # Plotting each method individually
        for i, (name, pred, color) in enumerate(methods):
            axes[i].plot(x_eval, pred, color=color, linewidth=2, label=name)
            if true_pdf is not None:
                axes[i].plot(x_eval, true_pdf(x_eval), 'r--', linewidth=2, label='True PDF')
            axes[i].scatter(self.data.flatten(), np.zeros_like(self.data.flatten()),
                            alpha=0.4, s=10, c='orange', label='Data')
            axes[i].set_title(name)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Combined comparison plot
        axes[3].plot(x_eval, knn_pred, 'b-', linewidth=2, label='kn-NN Neural Network')
        axes[3].plot(x_eval, parzen_pred, 'g-', linewidth=2, label='Parzen Window')
        axes[3].plot(x_eval, hist_pred, 'm-', linewidth=2, label='Histogram')
        if true_pdf is not None:
            axes[3].plot(x_eval, true_pdf(x_eval), 'r--', linewidth=2, label='True PDF')
        axes[3].scatter(self.data.flatten(), np.zeros_like(self.data.flatten()),
                        alpha=0.4, s=10, c='orange', label='Data')
        axes[3].set_title('All Methods Comparison')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('Density')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Printing quantitative comparison if we have the true PDF
        if true_pdf is not None:
            print("\nQuantitative Comparison:")
            print("-" * 50)

            evaluator = ModelEvaluator()

            def knn_pdf(x):
                return knn_model.predict(x.reshape(-1, 1) if x.ndim == 1 else x)

            def parzen_pdf(x):
                return np.exp(self.fitted_models['parzen']['model'].score_samples(x.reshape(-1, 1)))

            def hist_pdf(x):
                return self.fitted_models['histogram']['model'](x)

            methods_eval = [
                ('kn-NN Neural Network', knn_pdf),
                ('Parzen Window', parzen_pdf),
                ('Histogram', hist_pdf)
            ]

            for name, pdf_func in methods_eval:
                ise = evaluator.integrated_squared_error(pdf_func, true_pdf, x_range)
                mae = evaluator.mean_absolute_error(pdf_func, true_pdf, x_range)
                kl_div = evaluator.kullback_leibler_divergence(pdf_func, true_pdf, x_range)
                integral = evaluator.check_normalization(pdf_func, x_range)

                print(f"{name}:")
                print(f"  ISE: {ise:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  KL Divergence: {kl_div:.6f}")
                print(f"  Integral: {integral:.6f}")
                print()

        return methods


class HyperparameterTuning:
    # Automated hyperparameter tuning for the kn-NN Neural Network

    def __init__(self, data, true_pdf=None, x_range=None):
        self.data = data
        self.true_pdf = true_pdf
        self.x_range = x_range if x_range else (data.min() - 1, data.max() + 1)

    def tune_k1_parameter(self, k1_values=None, architecture=(50, 30),
                          max_iter=500, biased=False, verbose=True):
        # Trying different k1 values to find the best one
        # k1 controls how many neighbors we use

        if k1_values is None:
            k1_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]

        results = {}
        evaluator = ModelEvaluator()

        if verbose:
            print("Tuning k1 parameter...")
            print("-" * 40)

        for k1 in k1_values:
            from knn_neural_network import KnNNeuralNetwork

            model = KnNNeuralNetwork(
                k1=k1,
                architecture=architecture,
                max_iter=max_iter,
                random_state=42
            )

            try:
                model.fit(self.data, biased=biased, verbose=False)

                def model_pdf(x):
                    return model.predict(x.reshape(-1, 1) if x.ndim == 1 else x)

                if self.true_pdf is not None:
                    ise = evaluator.integrated_squared_error(model_pdf, self.true_pdf, self.x_range)
                    mae = evaluator.mean_absolute_error(model_pdf, self.true_pdf, self.x_range)
                    kl_div = evaluator.kullback_leibler_divergence(model_pdf, self.true_pdf, self.x_range)
                else:
                    # If we don't have true PDF, we use training loss as proxy
                    ise = model.mlp.loss_
                    mae = model.mlp.loss_
                    kl_div = model.mlp.loss_

                integral = evaluator.check_normalization(model_pdf, self.x_range)

                results[k1] = {
                    'model': model,
                    'ise': ise,
                    'mae': mae,
                    'kl_div': kl_div,
                    'integral': integral,
                    'training_loss': model.mlp.loss_
                }

                if verbose:
                    print(f"k1={k1:.1f}: ISE={ise:.6f}, MAE={mae:.6f}, Integral={integral:.3f}")

            except Exception as e:
                if verbose:
                    print(f"k1={k1:.1f}: Failed - {str(e)}")
                results[k1] = None

        # Finding the best k1 based on ISE
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_k1 = min(valid_results.keys(), key=lambda k: valid_results[k]['ise'])
            if verbose:
                print(f"\nBest k1: {best_k1} (ISE: {valid_results[best_k1]['ise']:.6f})")
        else:
            best_k1 = k1_values[len(k1_values) // 2]  # Default to middle value
            if verbose:
                print(f"\nNo valid results, using default k1: {best_k1}")

        return results, best_k1

    def tune_architecture(self, architectures=None, k1=1.0, max_iter=500,
                          biased=False, verbose=True):
        # Trying different neural network architectures

        if architectures is None:
            architectures = [
                (20,),  # Single layer, small
                (50,),  # Single layer, medium
                (30, 20),  # Two layers, small
                (50, 30),  # Two layers, medium - this seems to work well
                (100, 50),  # Two layers, large
                (50, 30, 20),  # Three layers
                (100, 50, 25),  # Three layers, large
                (30, 30, 30)  # Three layers, uniform
            ]

        results = {}
        evaluator = ModelEvaluator()

        if verbose:
            print("Tuning architecture...")
            print("-" * 40)

        for arch in architectures:
            from knn_neural_network import KnNNeuralNetwork

            model = KnNNeuralNetwork(
                k1=k1,
                architecture=arch,
                max_iter=max_iter,
                random_state=42
            )

            try:
                model.fit(self.data, biased=biased, verbose=False)

                def model_pdf(x):
                    return model.predict(x.reshape(-1, 1) if x.ndim == 1 else x)

                if self.true_pdf is not None:
                    ise = evaluator.integrated_squared_error(model_pdf, self.true_pdf, self.x_range)
                    mae = evaluator.mean_absolute_error(model_pdf, self.true_pdf, self.x_range)
                    kl_div = evaluator.kullback_leibler_divergence(model_pdf, self.true_pdf, self.x_range)
                else:
                    ise = model.mlp.loss_
                    mae = model.mlp.loss_
                    kl_div = model.mlp.loss_

                integral = evaluator.check_normalization(model_pdf, self.x_range)

                # Counting parameters to see model complexity
                n_params = sum([np.prod(layer.shape) for layer in model.mlp.coefs_]) + \
                           sum([layer.shape[0] for layer in model.mlp.intercepts_])

                results[arch] = {
                    'model': model,
                    'ise': ise,
                    'mae': mae,
                    'kl_div': kl_div,
                    'integral': integral,
                    'training_loss': model.mlp.loss_,
                    'n_parameters': n_params
                }

                if verbose:
                    print(f"{str(arch):15}: ISE={ise:.6f}, Parameters={n_params}")

            except Exception as e:
                if verbose:
                    print(f"{str(arch):15}: Failed - {str(e)}")
                results[arch] = None

        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_arch = min(valid_results.keys(), key=lambda k: valid_results[k]['ise'])
            if verbose:
                print(f"\nBest architecture: {best_arch} (ISE: {valid_results[best_arch]['ise']:.6f})")
        else:
            best_arch = (50, 30)  # Default architecture
            if verbose:
                print(f"\nNo valid results, using default architecture: {best_arch}")

        return results, best_arch

    def comprehensive_tuning(self, verbose=True):
        # Doing a comprehensive search over both k1 and architecture

        if verbose:
            print("=" * 60)
            print("Comprehensive Hyperparameter Tuning")
            print("=" * 60)

        # Phase 1: Tuning k1 with default architecture
        k1_results, best_k1 = self.tune_k1_parameter(verbose=verbose)

        if verbose:
            print(f"\nPhase 1 complete. Best k1: {best_k1}")
            print("\n" + "=" * 60)

        # Phase 2: Tuning architecture with best k1
        arch_results, best_arch = self.tune_architecture(k1=best_k1, verbose=verbose)

        if verbose:
            print(f"\nPhase 2 complete. Best architecture: {best_arch}")
            print("\n" + "=" * 60)

        # Phase 3: Fine-tuning k1 around the best value
        k1_fine_values = np.linspace(max(0.1, best_k1 - 0.5), best_k1 + 0.5, 11)
        k1_fine_results, best_k1_fine = self.tune_k1_parameter(
            k1_values=k1_fine_values,
            architecture=best_arch,
            verbose=verbose
        )

        if verbose:
            print(f"\nFinal tuning complete.")
            print(f"Optimal parameters: k1={best_k1_fine}, architecture={best_arch}")

        return {
            'best_k1': best_k1_fine,
            'best_architecture': best_arch,
            'k1_results': k1_results,
            'arch_results': arch_results,
            'k1_fine_results': k1_fine_results
        }


class ExperimentRunner:

    def __init__(self, data, true_pdf=None, x_range=None):
        self.data = data
        self.true_pdf = true_pdf
        self.x_range = x_range if x_range else (data.min() - 1, data.max() + 1)

    def run_sample_size_experiment(self, sample_sizes=None, k1=1.0,
                                   architecture=(50, 30), n_runs=5):
        # Testing how performance changes with different sample sizes

        if sample_sizes is None:
            sample_sizes = [50, 100, 200, 400, 800]

        results = {}
        evaluator = ModelEvaluator()

        print("Sample Size Experiment")
        print("-" * 40)

        for n in sample_sizes:
            if n > len(self.data):
                continue

            run_results = []

            # Doing multiple runs to get statistical significance
            for run in range(n_runs):
                np.random.seed(run)  # Different seed for each run
                indices = np.random.choice(len(self.data), n, replace=False)
                subsample = self.data[indices]

                from knn_neural_network import KnNNeuralNetwork

                model = KnNNeuralNetwork(
                    k1=k1,
                    architecture=architecture,
                    max_iter=500,
                    random_state=run
                )

                try:
                    model.fit(subsample, biased=False, verbose=False)

                    def model_pdf(x):
                        return model.predict(x.reshape(-1, 1) if x.ndim == 1 else x)

                    if self.true_pdf is not None:
                        ise = evaluator.integrated_squared_error(model_pdf, self.true_pdf, self.x_range)
                        mae = evaluator.mean_absolute_error(model_pdf, self.true_pdf, self.x_range)
                    else:
                        ise = model.mlp.loss_
                        mae = model.mlp.loss_

                    run_results.append({
                        'ise': ise,
                        'mae': mae,
                        'training_loss': model.mlp.loss_
                    })

                except Exception as e:
                    print(f"Failed for n={n}, run={run}: {str(e)}")

            # Calculating statistics across runs
            if run_results:
                ise_mean = np.mean([r['ise'] for r in run_results])
                ise_std = np.std([r['ise'] for r in run_results])
                mae_mean = np.mean([r['mae'] for r in run_results])
                mae_std = np.std([r['mae'] for r in run_results])

                results[n] = {
                    'ise_mean': ise_mean,
                    'ise_std': ise_std,
                    'mae_mean': mae_mean,
                    'mae_std': mae_std,
                    'raw_results': run_results
                }

                print(f"n={n:3d}: ISE={ise_mean:.6f}±{ise_std:.6f}, MAE={mae_mean:.6f}±{mae_std:.6f}")

        return results

    def generate_final_report(self, model, comparison_methods=None):
        # Generating a comprehensive report on the model performance

        print("=" * 80)
        print("FINAL EVALUATION REPORT")
        print("=" * 80)

        print(f"Model Configuration:")
        print(f"  k1 parameter: {model.k1}")
        print(f"  Architecture: {model.architecture}")
        print(f"  Training samples: {len(model.training_data)}")
        print(f"  kn value: {model.kn}")
        print(f"  Training iterations: {model.mlp.n_iter_}")
        print(f"  Final training loss: {model.mlp.loss_:.6f}")

        # Checking model properties
        x_eval = np.linspace(self.x_range[0], self.x_range[1], 1000)
        density_estimates = model.predict(x_eval)

        evaluator = ModelEvaluator()
        integral = evaluator.check_normalization(lambda x: model.predict(x.reshape(-1, 1)), self.x_range)

        print(f"\nModel Validation:")
        print(f"  Integral of estimated PDF: {integral:.6f}")
        print(f"  Non-negative outputs: {np.all(density_estimates >= 0)}")
        print(f"  Output range: [{density_estimates.min():.6f}, {density_estimates.max():.6f}]")

        # Comparing with true PDF if available
        if self.true_pdf is not None:
            def model_pdf(x):
                return model.predict(x.reshape(-1, 1) if x.ndim == 1 else x)

            ise = evaluator.integrated_squared_error(model_pdf, self.true_pdf, self.x_range)
            mae = evaluator.mean_absolute_error(model_pdf, self.true_pdf, self.x_range)
            kl_div = evaluator.kullback_leibler_divergence(model_pdf, self.true_pdf, self.x_range)

            print(f"\nComparison with True PDF:")
            print(f"  Integrated Squared Error: {ise:.6f}")
            print(f"  Mean Absolute Error: {mae:.6f}")
            print(f"  KL Divergence: {kl_div:.6f}")

        if comparison_methods is not None:
            print(f"\nBaseline Comparison:")
            baseline = BaselineComparison(self.data)
            baseline.compare_all_methods(model, self.true_pdf, self.x_range, figsize=(15, 8))

        print("\n" + "=" * 80)
        print("Report generation complete.")


def save_model_results(model, filename, additional_data=None):
    # Saving model
    import pickle

    save_data = {
        'model_params': {
            'k1': model.k1,
            'architecture': model.architecture,
            'max_iter': model.max_iter,
            'learning_rate': model.learning_rate,
            'random_state': model.random_state
        },
        'training_data': model.training_data,
        'targets': model.targets,
        'kn': model.kn,
        'training_history': model.training_history,
        'scaler': model.scaler,
        'mlp_params': {
            'coefs_': model.mlp.coefs_,
            'intercepts_': model.mlp.intercepts_,
            'loss_': model.mlp.loss_,
            'n_iter_': model.mlp.n_iter_
        }
    }

    if additional_data:
        save_data.update(additional_data)

    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Model saved to {filename}")


def load_model_results(filename):
    # Loading previously saved model results
    import pickle

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    print(f"Model loaded from {filename}")
    return data


def evaluate_on_instructor_dataset(data_file_path, model_params=None):
    # Evaluate of the algorithm on the instructor's dataset

    try:
        instructor_data = np.loadtxt(data_file_path)
        print(f"Loaded {len(instructor_data)} samples from instructor's dataset")
        print(f"Data range: [{instructor_data.min():.3f}, {instructor_data.max():.3f}]")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    if model_params is None:
        model_params = {
            'k1': 1.0,
            'architecture': (50, 30),
            'max_iter': 1000,
            'learning_rate': 0.001
        }

    from knn_neural_network import KnNNeuralNetwork

    print("\nPerforming hyperparameter tuning...")
    tuner = HyperparameterTuning(instructor_data)
    tuning_results = tuner.comprehensive_tuning()

    # Training the final model with optimal parameters
    print("\nTraining final model...")
    final_model = KnNNeuralNetwork(
        k1=tuning_results['best_k1'],
        architecture=tuning_results['best_architecture'],
        max_iter=1000,
        learning_rate=0.001,
        random_state=42
    )

    print("Training unbiased version...")
    final_model.fit(instructor_data, biased=False, validation_split=0.2, verbose=True)

    # Also training biased version for comparison
    print("Training biased version for comparison...")
    biased_model = KnNNeuralNetwork(
        k1=tuning_results['best_k1'],
        architecture=tuning_results['best_architecture'],
        max_iter=1000,
        learning_rate=0.001,
        random_state=42
    )
    biased_model.fit(instructor_data, biased=True, validation_split=0.2, verbose=True)

    experiment_runner = ExperimentRunner(instructor_data)
    experiment_runner.generate_final_report(final_model, comparison_methods=True)

    print("\nGenerating visualizations...")

    # Comparing biased vs unbiased
    final_model.compare_biased_unbiased(instructor_data)

    final_model.plot_training_curves()

    final_model.plot_density_estimate(
        title="Final kn-NN Density Estimation on Instructor's Dataset"
    )

    save_model_results(
        final_model,
        'instructor_dataset_results.pkl',
        {
            'instructor_data': instructor_data,
            'tuning_results': tuning_results,
            'biased_model_loss': biased_model.mlp.loss_
        }
    )

    print("\nEvaluation complete! Results saved to 'instructor_dataset_results.pkl'")

    return final_model, tuning_results