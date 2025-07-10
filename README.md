# kn-NN Neural Network for Density Estimation

This is our implementation of a kn-nearest neighbor neural network for probability density function estimation. Instead of using traditional Parzen windows, we are using kn-NN to generate training targets for a multilayer perceptron. This was inspired by the Parzen Neural Networks paper we discussed in class, but with some modifications to make it work better.

## What's in here

```
knn_neural_network/
├── knn_neural_network.py    # Main implementation of the algorithm
├── evaluation_utils.py      # Evaluation metrics and comparison tools
├── demo_script.py          # Demo script with different experiments
├── README.md             
└── requirements.txt        # Required packages
```

## How it works

Basic idea: instead of using the neural network directly for density estimation, we use kn-NN to generate "pseudo-targets" and then train a neural network to learn this mapping.

### The algorithm:

1. **Target generation**: For each training point xi, we calculate a density estimate using the kn-nearest neighbor method
2. **Neural network training**: We train an MLP to learn the mapping from input points to these density estimates
3. **Density estimation**: We use the trained network to estimate density at any new point

### Biased vs Unbiased

- **Biased**: When generating the target for point xi, include xi itself in the kn-NN calculation
- **Unbiased**: When generating the target for point xi, exclude xi from the kn-NN calculation (leave-one-out)

The unbiased version is usually better and is what the paper recommends.

## Main Classes

### KnNNeuralNetwork

This is the main class that implements the algorithm.

**Key parameters:**
- `k1`: Controls how many neighbors to use (kn = k1 * sqrt(n))
- `architecture`: Tuple of hidden layer sizes, like (50, 30)
- `max_iter`: Maximum training iterations
- `learning_rate`: Learning rate for the neural network

**Main methods:**
- `fit(X, biased=False)`: Trains the model
- `predict(X)`: Gets density estimates
- `plot_density_estimate()`: Visualizes the results
- `compare_biased_unbiased()`: Compares both versions side by side

### Other useful classes

- **ModelEvaluator**: Calculates metrics like ISE, MAE, KL divergence
- **BaselineComparison**: Compares with Parzen window and histogram methods
- **HyperparameterTuning**: Automatically finds good parameters
- **ExperimentRunner**: Runs various experiments and generate reports

### Validation:
- Built-in checks for non-negative outputs
- Monitoring of PDF normalization during training
- Training curve visualization for convergence checking

## References

This implementation is based on:
- The Parzen Neural Networks paper (Trentin, 2019)
- Standard kn-NN density estimation theory
- Various neural network optimization techniques I learned in class
