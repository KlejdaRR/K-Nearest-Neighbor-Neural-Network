# kn-NN Neural Network for Density Estimation

This project implements a hybrid approach to probability density function (PDF) estimation by combining k-nearest neighbor (k-NN) methods with neural networks. Inspired by Parzen Neural Networks, we replace traditional Parzen windows with k-NN to generate training targets for a multilayer perceptron (MLP), achieving robust density estimation on real-world data like the Old Faithful geyser dataset and a challenging dataset provided from our professor.

## Key Features
- **k-NN + Neural Network Hybrid**: This project Uses k-NN density estimates as pseudo-targets to train a Multi Layered Perceptron.
- **Biased/Unbiased Estimation**: It implements both versions for comparison (unbiased recommended).
- **Comprehensive Evaluation**: Includes comparison with KDE, histograms, and quantitative metrics.
- **Real-World Validation**: We demonstrate this on the Old Faithful dataset (eruption durations and waiting times).
- **Advanced Regularization**: We have included L2 regularization, target scaling, and boundary damping for stable training due to the challenges proposed by the professor's dataset.
- **Automatic Hyperparameter Tuning**: Built-in grid search for optimal k1 and architecture selection.

## What's in here
```
knn_neural_network/
├── datasets/
│   └── faithful.dat.txt              # Old Faithful dataset
├── knn_neural_network.py             # Main algorithm implementation
├── evaluation_utils.py               # Metrics, baseline comparisons, hyperparameter tuning
├── demo_script.py                    # Preconfigured experiments (Old Faithful dataset)
├── simple_faithful_example.py       # Minimal example with Old Faithful data
├── professor_dataset.py             # Analysis script for professor's dataset
├── README.md
└── requirements.txt                  # Dependencies
```

## How It Works
1. **Target Generation**:  
   For each point `x_i`, we compute a k-NN density estimate (`kn = k1 * sqrt(n)`).  
   - **Unbiased (default)**: Excludes `x_i` (leave-one-out).  
   - **Biased**: Includes `x_i` (may overfit).  

2. **Neural Network Training**:  
   A Multi Layered Perceptron learns to map input points → k-NN density estimates.  

3. **Density Estimation**:  
   The trained MLP predicts densities for new points.  

## Performance Results

### Professor's Dataset Analysis
Our improved implementation shows very good performance on this challenging dataset:

- **Cross-Validation Log-Likelihood**: -0.108568 ± 0.154158
- **Baseline Comparison**:
  - Parzen Window: -1.101025 ± 0.122182
  - Histogram: -1.357119 ± 0.532375
- **Proper normalization**: ∫PDF ≈ 3.0
- **Stable convergence**: 864 iterations, final loss 0.384939

### Old Faithful Dataset Results
The model successfully captures the distributions of:
- **Eruption Durations**: Peaks at ~2 and ~4.5 minutes.
- **Waiting Times**: Peaks at ~55 and ~80 minutes.  

**Key findings:**
- Strong correlation (r = 0.901) between eruption duration and waiting time.
- Unbiased estimates are smoother and more reliable than biased ones.
- Competitive performance with Kernel Density Estimation (KDE).  

## Main Classes

### KnNNeuralNetwork
**Key Parameters:**
- `k1`: This parameter controls neighbor count (kn = k1 * sqrt(n))
- `architecture`: Multi Layer Perceptron's hidden layers (e.g., (100, 50))
- `max_iter`: Training iterations (default: 2000)
- `learning_rate`: Learning rate (default: 0.0001)
- `alpha`: L2 regularization (default: 0.001)

**Methods:**
- `fit(X, biased=False)`: Train on data X
- `predict(X)`: Return density estimates
- `plot_density_estimate()`: Visualize PDF
- `compare_biased_unbiased()`: Compare estimator types
- `plot_training_curves()`: Monitor convergence

### Support Classes
- **ModelEvaluator**: Metrics (log-likelihood, normalization checks)
- **BaselineComparison**: Comparing with KDE/histograms
- **HyperparameterTuning**: Optimizing k1 and architectures
- **ExperimentRunner**: Running systematic experiments

## Validation

### Mathematical Properties
- **PDF Normalization**: Ensuring ∫PDF ≈ 1
- **Non-negativity**: All density estimates are ≥ 0
- **Smoothness**: Continuous density functions
- **Boundary Behavior**: Proper decay outside data range

### Performance Metrics
- **Cross-validation log-likelihood**: Primary evaluation metric
- **Training curves**: Monitoring of convergence/overfitting
- **Baseline comparison**: Performance vs KDE and histograms
- **Hyperparameter sensitivity**: Robustness across parameter ranges

## Installation

```bash

# Install dependencies
pip install -r requirements.txt

# Run basic demo
python simple_faithful_example.py

# Run professor's dataset evaluation
python professor_dataset.py

```

## References
- Trentin, E. (2019). Parzen Neural Networks. 
- Old Faithful Dataset: R Statistical Software Manual
- Scikit-learn: Machine Learning in Python

## Authors
**Klejda Rrapaj** (k.rrapaj@student.unisi.it)  
**Sildi Ricku** (s.ricku@student.unisi.it)
