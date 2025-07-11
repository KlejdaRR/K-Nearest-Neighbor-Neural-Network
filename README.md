# kn-NN Neural Network for Density Estimation

This project implements a hybrid approach to probability density function (PDF) estimation by combining k-nearest neighbor (k-NN) methods with neural networks. Inspired by Parzen Neural Networks, we replace traditional Parzen windows with k-NN to generate training targets for a multilayer perceptron (MLP), achieving robust density estimation on real-world data like the Old Faithful geyser dataset.

## Key Features
- **k-NN + Neural Network Hybrid**: Uses k-NN density estimates as pseudo-targets to train an MLP.
- **Biased/Unbiased Estimation**: Implements both versions for comparison (unbiased recommended).
- **Comprehensive Evaluation**: Includes comparison with KDE, histograms, and quantitative metrics.
- **Real-World Validation**: Demonstrated on the Old Faithful dataset (eruption durations and waiting times).

## What's in here
knn_neural_network/
|── datasets/|──────────── faithful.dat.txt # dataset
├── knn_neural_network.py # Main algorithm implementation
├── evaluation_utils.py # Metrics, baseline comparisons, hyperparameter tuning
├── demo_script.py # Preconfigured experiments (Old Faithful dataset)
├── simple_faithful_example.py # Minimal example with Old Faithful data
├── README.md
└── requirements.txt # Dependencies

## How It Works
1. **Target Generation**:  
   For each point `x_i`, compute a k-NN density estimate (`kn = k1 * sqrt(n)`).  
   - **Unbiased (default)**: Excludes `x_i` (leave-one-out).  
   - **Biased**: Includes `x_i` (may overfit).  

2. **Neural Network Training**:  
   An MLP learns to map input points → k-NN density estimates.  

3. **Density Estimation**:  
   The trained MLP predicts densities for new points.  

## Old Faithful Dataset Results
The model successfully captures the bimodal distributions of:
- **Eruption Durations**: Peaks at ~2 and ~4.5 minutes.
- **Waiting Times**: Peaks at ~55 and ~80 minutes.  

## Key findings:
- Strong correlation (r = 0.901) between eruption duration and waiting time.
- Unbiased estimates are smoother and more reliable than biased ones.
- Competitive performance with Kernel Density Estimation (KDE).  

## Demo Scripts
Full Analysis:

bash
python demo_script.py --full_analysis
Hyperparameter Tuning:

bash
python demo_script.py --hyperparameter_tuning

## Main Classes
## KnNNeuralNetwork
Key Parameters:

k1: Controls neighbor count (kn = k1 * sqrt(n)).
architecture: MLP hidden layers (e.g., (50, 30)).
max_iter: Training iterations.

## Methods:

fit(X, biased=False): Train on data X.
predict(X): Return density estimates.
plot_density_estimate(): Visualize PDF.
compare_biased_unbiased(): Compare estimator types.

## Support Classes
ModelEvaluator: Metrics (log-likelihood, normalization checks).
BaselineComparison: Compare with KDE/histograms.
HyperparameterTuning: Optimize k1 and architectures.

## Validation
Training Curves: Monitor convergence/overfitting (Figure 1).
PDF Normalization: Ensure ∫PDF ≈ 1.
Quantitative Metrics: Cross-validation log-likelihood, comparison with baselines.

## References
Trentin (2019). Parzen Neural Networks.
Old Faithful Dataset: R Manual.

## Authors:
Klejda Rrapaj (k.rrapaj@student.unisi.it),
Sildi Ricku (s.ricku@student.unisi.it)
