import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


class ModelEvaluator:
    # Helper class for evaluating density estimation models
    # We needed this to compare our results with the true PDF
    @staticmethod
    def cross_validation_score(model_func, data, cv_folds=5):
        # Cross-validation score for density estimation using log-likelihood
        # Since we don't have true PDF for real data, we use CV log-likelihood

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in kf.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]

            # Train model on training fold
            try:
                model = model_func(train_data)
                # Evaluate on test fold
                test_densities = model.predict(test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data)
                # Log-likelihood (avoiding log(0))
                test_densities = np.maximum(test_densities, 1e-10)
                log_likelihood = np.mean(np.log(test_densities))
                scores.append(log_likelihood)
            except:
                scores.append(-np.inf)

        return np.mean(scores), np.std(scores)

    @staticmethod
    def check_normalization(pdf_func, x_range, n_points=10000):
        # Checking if the PDF integrates to 1 (as it should)
        # This helped us debug issues with our implementation
        x_eval = np.linspace(x_range[0], x_range[1], n_points)
        pdf_vals = pdf_func(x_eval)
        integral = np.trapz(pdf_vals, x_eval)
        return integral

    @staticmethod
    def compute_bandwidth_comparison(data, bandwidths=None):
        # Comparing different bandwidths for KDE using cross-validation
        if bandwidths is None:
            bandwidths = np.logspace(-2, 1, 20)

        data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=5, scoring='score')
        grid.fit(data_reshaped)
        return grid.best_params_['bandwidth'], grid.best_score_


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
            bandwidth, _ = ModelEvaluator.compute_bandwidth_comparison(self.data.flatten())

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
            bins = int(np.log2(len(self.data)) + 1)

        hist, bin_edges = np.histogram(self.data.flatten(), bins=bins, density=True)

        def histogram_pdf(x):
            # Converting histogram to a function so we can evaluate anywhere
            x = np.atleast_1d(x)
            result = np.zeros_like(x)
            for i in range(len(bin_edges) - 1):
                mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
                result[mask] = hist[i]
            # Handling the last bin edge
            mask = x == bin_edges[-1]
            result[mask] = hist[-1]
            return result

        self.fitted_models['histogram'] = {
            'model': histogram_pdf,
            'bins': bins,
            'name': f'Histogram ({bins} bins)'
        }
        return histogram_pdf

    def compare_all_methods(self, knn_model, x_range=None, n_points=1000, figsize=(15, 10)):
        # Comparing all methods side by side with quantitative metrics
        if x_range is None:
            data_flat = self.data.flatten()
            margin = (data_flat.max() - data_flat.min()) * 0.1
            x_range = (data_flat.min() - margin, data_flat.max() + margin)

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
        axes[3].scatter(self.data.flatten(), np.zeros_like(self.data.flatten()),
                        alpha=0.4, s=10, c='orange', label='Data')
        axes[3].set_title('All Methods Comparison')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('Density')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Quantitative comparison using cross-validation log-likelihood
        print("\nQuantitative Comparison (Cross-Validation Log-Likelihood):")
        print("-" * 60)
        evaluator = ModelEvaluator()

        def knn_model_func(train_data):
            from knn_neural_network import KnNNeuralNetwork
            temp_model = KnNNeuralNetwork(
                k1=knn_model.k1,
                architecture=knn_model.architecture,
                max_iter=knn_model.max_iter,
                learning_rate=knn_model.learning_rate,
                random_state=42
            )
            temp_model.fit(train_data, biased=False, verbose=False)
            return temp_model

        def parzen_model_func(train_data):
            bandwidth, _ = evaluator.compute_bandwidth_comparison(train_data)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(train_data.reshape(-1, 1))

            class KDEWrapper:
                def __init__(self, kde_model):
                    self.kde = kde_model

                def predict(self, x):
                    return np.exp(self.kde.score_samples(x.reshape(-1, 1) if x.ndim == 1 else x))

            return KDEWrapper(kde)

        def hist_model_func(train_data):
            bins = int(np.log2(len(train_data)) + 1)
            hist, bin_edges = np.histogram(train_data, bins=bins, density=True)

            def histogram_pdf(x):
                x = np.atleast_1d(x)
                result = np.zeros_like(x)
                for i in range(len(bin_edges) - 1):
                    mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
                    result[mask] = hist[i]
                mask = x == bin_edges[-1]
                result[mask] = hist[-1]
                return result

            class HistWrapper:
                def predict(self, x):
                    return histogram_pdf(x.flatten() if x.ndim > 1 else x)

            return HistWrapper()

        methods_eval = [
            ('kn-NN Neural Network', knn_model_func),
            ('Parzen Window', parzen_model_func),
            ('Histogram', hist_model_func)
        ]

        data_flat = self.data.flatten()
        for name, model_func in methods_eval:
            try:
                cv_mean, cv_std = evaluator.cross_validation_score(model_func, data_flat)
                print(f"{name}:")
                print(f"  CV Log-Likelihood: {cv_mean:.6f} ± {cv_std:.6f}")

                # Also check normalization
                if name == 'kn-NN Neural Network':
                    def knn_pdf(x):
                        return knn_model.predict(x.reshape(-1, 1) if x.ndim == 1 else x)

                    integral = evaluator.check_normalization(knn_pdf, x_range)
                    print(f"  Normalization: {integral:.6f}")

                print()
            except Exception as e:
                print(f"{name}: Error in evaluation - {str(e)}")

        return methods


class HyperparameterTuning:
    # Automated hyperparameter tuning for the kn-NN Neural Network

    def __init__(self, data, x_range=None):
        self.data = data
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
            def model_func(train_data):
                from knn_neural_network import KnNNeuralNetwork
                model = KnNNeuralNetwork(
                    k1=k1,
                    architecture=architecture,
                    max_iter=max_iter,
                    random_state=42
                )
                model.fit(train_data, biased=biased, verbose=False)
                return model

            try:
                cv_mean, cv_std = evaluator.cross_validation_score(model_func, self.data)

                # Also training a full model for other metrics
                full_model = model_func(self.data)

                def model_pdf(x):
                    return full_model.predict(x.reshape(-1, 1) if x.ndim == 1 else x)

                integral = evaluator.check_normalization(model_pdf, self.x_range)

                results[k1] = {
                    'model': full_model,
                    'cv_score': cv_mean,
                    'cv_std': cv_std,
                    'integral': integral,
                    'training_loss': full_model.mlp.loss_
                }

                if verbose:
                    print(f"k1={k1:.1f}: CV Score={cv_mean:.6f}±{cv_std:.6f}, Integral={integral:.3f}")

            except Exception as e:
                if verbose:
                    print(f"k1={k1:.1f}: Failed - {str(e)}")
                results[k1] = None

        # Finding the best k1 based on CV score (higher is better for log-likelihood)
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_k1 = max(valid_results.keys(), key=lambda k: valid_results[k]['cv_score'])
            if verbose:
                print(f"\nBest k1: {best_k1} (CV Score: {valid_results[best_k1]['cv_score']:.6f})")
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
                (50, 30),  # Two layers, medium
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
            def model_func(train_data):
                from knn_neural_network import KnNNeuralNetwork
                model = KnNNeuralNetwork(
                    k1=k1,
                    architecture=arch,
                    max_iter=max_iter,
                    random_state=42
                )
                model.fit(train_data, biased=biased, verbose=False)
                return model

            try:
                cv_mean, cv_std = evaluator.cross_validation_score(model_func, self.data)

                # Training full model for other metrics
                full_model = model_func(self.data)

                def model_pdf(x):
                    return full_model.predict(x.reshape(-1, 1) if x.ndim == 1 else x)

                integral = evaluator.check_normalization(model_pdf, self.x_range)

                # Counting parameters
                n_params = sum([np.prod(layer.shape) for layer in full_model.mlp.coefs_]) + \
                           sum([layer.shape[0] for layer in full_model.mlp.intercepts_])

                results[arch] = {
                    'model': full_model,
                    'cv_score': cv_mean,
                    'cv_std': cv_std,
                    'integral': integral,
                    'training_loss': full_model.mlp.loss_,
                    'n_parameters': n_params
                }

                if verbose:
                    print(f"{str(arch):15}: CV Score={cv_mean:.6f}±{cv_std:.6f}, Parameters={n_params}")

            except Exception as e:
                if verbose:
                    print(f"{str(arch):15}: Failed - {str(e)}")
                results[arch] = None

        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_arch = max(valid_results.keys(), key=lambda k: valid_results[k]['cv_score'])
            if verbose:
                print(f"\nBest architecture: {best_arch} (CV Score: {valid_results[best_arch]['cv_score']:.6f})")
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
    # Running experiments on real data

    def __init__(self, data, x_range=None):
        self.data = data
        self.x_range = x_range if x_range else (data.min() - 1, data.max() + 1)

    def run_sample_size_experiment(self, sample_sizes=None, k1=1.0,
                                   architecture=(50, 30), n_runs=5):
        # Testing how performance changes with different sample sizes
        if sample_sizes is None:
            max_size = len(self.data)
            sample_sizes = [30, 50, 100, 150, 200, int(max_size * 0.8)]
            sample_sizes = [s for s in sample_sizes if s < max_size]

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

                def model_func(train_data):
                    from knn_neural_network import KnNNeuralNetwork
                    model = KnNNeuralNetwork(
                        k1=k1,
                        architecture=architecture,
                        max_iter=500,
                        random_state=run
                    )
                    model.fit(train_data, biased=False, verbose=False)
                    return model

                try:
                    cv_score, cv_std = evaluator.cross_validation_score(model_func, subsample)

                    # Also getting training loss
                    full_model = model_func(subsample)
                    training_loss = full_model.mlp.loss_

                    run_results.append({
                        'cv_score': cv_score,
                        'training_loss': training_loss
                    })
                except Exception as e:
                    print(f"Failed for n={n}, run={run}: {str(e)}")

            # Calculating statistics across runs
            if run_results:
                cv_mean = np.mean([r['cv_score'] for r in run_results])
                cv_std = np.std([r['cv_score'] for r in run_results])
                loss_mean = np.mean([r['training_loss'] for r in run_results])
                loss_std = np.std([r['training_loss'] for r in run_results])

                results[n] = {
                    'cv_score_mean': cv_mean,
                    'cv_score_std': cv_std,
                    'ise_mean': loss_mean,  # For compatibility with plotting code
                    'ise_std': loss_std,  # For compatibility with plotting code
                    'raw_results': run_results
                }

                print(f"n={n:3d}: CV Score={cv_mean:.6f}±{cv_std:.6f}, Loss={loss_mean:.6f}±{loss_std:.6f}")

        return results

    def generate_final_report(self, model, comparison_methods=True):
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

        # Cross-validation evaluation
        def model_func(train_data):
            from knn_neural_network import KnNNeuralNetwork
            temp_model = KnNNeuralNetwork(
                k1=model.k1,
                architecture=model.architecture,
                max_iter=model.max_iter,
                learning_rate=model.learning_rate,
                random_state=42
            )
            temp_model.fit(train_data, biased=False, verbose=False)
            return temp_model

        cv_score, cv_std = evaluator.cross_validation_score(model_func, self.data)
        print(f"\nCross-Validation Performance:")
        print(f"  CV Log-Likelihood: {cv_score:.6f} ± {cv_std:.6f}")

        if comparison_methods:
            print(f"\nBaseline Comparison:")
            baseline = BaselineComparison(self.data)
            baseline.compare_all_methods(model, self.x_range, figsize=(15, 8))

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