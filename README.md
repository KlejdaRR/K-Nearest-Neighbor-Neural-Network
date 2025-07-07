# kn-NN Neural Network for Probability Density Function Estimation

This project implements a novel approach to probability density function estimation by combining k-nearest neighbor methods with neural networks. Instead of using Parzen windows like traditional Parzen Neural Networks (PNNs), this implementation uses kn-NN algorithms to generate synthetic training targets for multilayer perceptrons (MLPs).

## Project Structure

```
knn_neural_network/
├── knn_neural_network.py    # Main implementation
├── evaluation_utils.py      # Evaluation and comparison utilities
├── demo_script.py          # Demonstration script
├── README.md               # This file
└── requirements.txt        # Dependencies
```

## Installation

### Requirements

```bash
pip install numpy matplotlib scikit-learn scipy
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scikit-learn`: Neural networks and baseline methods
- `scipy`: Scientific computing utilities

## Quick Start

### Basic Usage

```python
from knn_neural_network import KnNNeuralNetwork, generate_test_data

# Generate test data
data, true_pdf = generate_test_data('mixture', n_samples=200)

# Create and train model
model = KnNNeuralNetwork(k1=1.0, architecture=(50, 30))
model.fit(data, biased=False, verbose=True)

# Visualize results
model.plot_density_estimate(true_pdf=true_pdf)
```

### Command Line Interface

```bash
# Basic demonstration
python demo_script.py

# Run on instructor's dataset
python demo_script.py --instructor_data data.txt

# Full experiments
python demo_script.py --full_experiments

# Hyperparameter tuning only
python demo_script.py --hyperparameter_tuning

# Custom parameters
python demo_script.py --distribution normal --n_samples 300 --k1 1.5
```

## Algorithm Overview

### Core Concept

The kn-NN Neural Network estimates probability density functions by:

1. **Target Generation**: For each training point xi, compute a density estimate using k-nearest neighbor method
2. **Neural Network Training**: Train an MLP to learn the mapping from input patterns to these density estimates
3. **Density Estimation**: Use the trained network to estimate density at any point

### Mathematical Foundation

The kn-NN density estimator is:

```
p(x₀) ≈ kn/n / V
```

Where:
- `kn = k₁ × √n` (number of nearest neighbors)
- `V` = volume of the ball containing kn nearest neighbors
- `n` = total number of training samples

### Biased vs Unbiased Versions

- **Biased**: Uses all training samples to generate targets
- **Unbiased**: When generating target for xi, excludes xi from the kn-NN calculation (recommended)

## Main Classes

### KnNNeuralNetwork

The main class implementing the kn-NN Neural Network.

#### Parameters:
- `k1`: Base parameter for kn calculation (default: 1.0)
- `architecture`: Tuple defining hidden layer sizes (default: (50, 30))
- `max_iter`: Maximum training iterations (default: 1000)
- `learning_rate`: Learning rate for neural network (default: 0.001)
- `random_state`: Random seed for reproducibility (default: 42)

#### Key Methods:
- `fit(X, biased=False)`: Train the model
- `predict(X)`: Predict density values
- `plot_density_estimate()`: Visualize the estimated density
- `compare_biased_unbiased()`: Compare biased and unbiased versions

### ModelEvaluator

Utility class for quantitative evaluation.

#### Metrics:
- **Integrated Squared Error (ISE)**: ∫(p_estimated(x) - p_true(x))² dx
- **Mean Absolute Error (MAE)**: Mean |p_estimated(x) - p_true(x)|
- **Kullback-Leibler Divergence**: D_KL(P||Q)
- **Normalization Check**: ∫p_estimated(x)dx ≈ 1

### BaselineComparison

Compare kn-NN with traditional methods:
- Parzen Window (Kernel Density Estimation)
- Histogram-based estimation

### HyperparameterTuning

Automated hyperparameter optimization:
- k1 parameter tuning
- Architecture optimization
- Comprehensive grid search

## Experimental Framework

### Available Experiments

1. **Basic Demonstration**: Quick start with synthetic data
2. **Biased vs Unbiased Comparison**: Compare both versions
3. **Hyperparameter Tuning**: Find optimal parameters
4. **Sample Size Analysis**: Performance vs dataset size
5. **Baseline Comparison**: Compare with traditional methods
6. **Multiple Distributions**: Test on various distributions

### Test Distributions

- **Mixture**: Mixture of two Gaussians
- **Normal**: Single Gaussian distribution  
- **Exponential**: Exponential distribution
- **Uniform**: Uniform distribution

## Usage Examples

### Example 1: Basic Training

```python
import numpy as np
from knn_neural_network import KnNNeuralNetwork

# Your data (1D array)
data = np.random.normal(0, 1, 500)  # Example data

# Create model
model = KnNNeuralNetwork(
    k1=1.0,
    architecture=(50, 30),
    max_iter=1000
)

# Train model (unbiased version)
model.fit(data, biased=False, validation_split=0.2)

# Predict density at new points
x_new = np.linspace(-3, 3, 100)
density_estimates = model.predict(x_new)

# Visualize
model.plot_density_estimate()
```

### Example 2: Hyperparameter Tuning

```python
from evaluation_utils import HyperparameterTuning

# Initialize tuner
tuner = HyperparameterTuning(data)

# Comprehensive tuning
results = tuner.comprehensive_tuning()

print(f"Best k1: {results['best_k1']}")
print(f"Best architecture: {results['best_architecture']}")
```

### Example 3: Comparison with Baselines

```python
from evaluation_utils import BaselineComparison

# Train your model
model = KnNNeuralNetwork(k1=1.0, architecture=(50, 30))
model.fit(data, biased=False)

# Compare with baselines
baseline = BaselineComparison(data)
baseline.compare_all_methods(model, true_pdf=true_pdf_function)
```

### Example 4: Evaluation on Instructor's Dataset

```python
from evaluation_utils import evaluate_on_instructor_dataset

# Evaluate on instructor's data
model, results = evaluate_on_instructor_dataset('instructor_data.txt')
```

## Advanced Features

### Model Persistence

```python
from evaluation_utils import save_model_results, load_model_results

# Save model
save_model_results(model, 'my_model.pkl')

# Load model
loaded_data = load_model_results('my_model.pkl')
```

### Custom Evaluation Metrics

```python
from evaluation_utils import ModelEvaluator

evaluator = ModelEvaluator()

# Calculate ISE
ise = evaluator.integrated_squared_error(
    estimated_pdf=lambda x: model.predict(x.reshape(-1, 1)),
    true_pdf=true_pdf_function,
    x_range=(data.min(), data.max())
)
```

## Performance Considerations

### Computational Complexity

- **Training**: O(n² + nEW³) where n=samples, E=epochs, W=parameters
- **Testing**: O(WT) where T=test samples (linear in test size)

### Memory Usage

- Stores training data and neural network parameters
- More memory efficient than traditional kn-NN during testing
- Scales better than Parzen Window methods

### Speed Optimization Tips

1. Use smaller architectures for faster training
2. Reduce max_iter for quicker experiments  
3. Use validation_split for early stopping
4. Consider parallel processing for hyperparameter tuning

## Project Validation

### Quality Checks

The implementation includes several validation mechanisms:

1. **Non-negative outputs**: Ensures density estimates are ≥ 0
2. **Normalization monitoring**: Tracks ∫p(x)dx during training
3. **Training curves**: Monitor convergence
4. **Cross-validation**: Built-in validation split

### Expected Results

For well-behaved 1D distributions:
- ISE should decrease with more training data
- Unbiased version typically outperforms biased
- Should outperform histogram for smooth distributions
- Competitive with Parzen Window methods

## Troubleshooting

### Common Issues

1. **Training fails to converge**:
   - Reduce learning rate
   - Increase max_iter
   - Try different architecture

2. **Poor density estimates**:
   - Tune k1 parameter
   - Try different architectures
   - Check data normalization

3. **Negative density values**:
   - Outputs are automatically clipped to ≥ 0
   - Consider different activation functions

4. **Memory issues**:
   - Reduce architecture size
   - Use smaller datasets for tuning
   - Process in batches

### Debug Mode

```python
# Enable verbose output
model.fit(data, biased=False, verbose=True)

# Plot training curves to check convergence
model.plot_training_curves()

# Check normalization
from evaluation_utils import ModelEvaluator
evaluator = ModelEvaluator()
integral = evaluator.check_normalization(
    lambda x: model.predict(x.reshape(-1, 1)),
    (data.min(), data.max())
)
print(f"Integral: {integral}")
```

## For the Course Project

### Instructor's Dataset Evaluation

When you receive the instructor's dataset:

```bash
# Place the data file in the project directory
# Run evaluation (this will perform hyperparameter tuning automatically)
python demo_script.py --instructor_data instructor_data.txt
```

This will:
1. Load the instructor's dataset
2. Perform comprehensive hyperparameter tuning
3. Train both biased and unbiased models
4. Generate comparison plots
5. Save results to `instructor_dataset_results.pkl`

### Expected Deliverables

The implementation provides everything needed for the course project:

1. ✅ **Algorithm Implementation**: Complete kn-NN Neural Network
2. ✅ **Biased/Unbiased Versions**: Both variants implemented
3. ✅ **1D Focus**: Optimized for one-dimensional data
4. ✅ **Hyperparameter Exploration**: Automated tuning framework
5. ✅ **Learning Curves**: Training progress visualization
6. ✅ **Comparison Plots**: Visual assessment capability
7. ✅ **Model Selection**: Built-in architecture optimization

## References

This implementation is based on the theoretical framework presented in the course materials, particularly:

- Parzen Neural Networks (Trentin, 2019)
- k-nearest neighbor density estimation theory
- Non-parametric density estimation methods
- Neural network optimization techniques

## License

This code is provided for educational purposes as part of the AI course project.
