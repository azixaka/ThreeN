# ThreeN Neural Network Library for .NET

[![NuGet Version](https://img.shields.io/nuget/v/ThreeN.svg)](https://www.nuget.org/packages/ThreeN/)
[![NuGet Downloads](https://img.shields.io/nuget/dt/ThreeN.svg)](https://www.nuget.org/packages/ThreeN/)
[![Build Status](https://github.com/azixaka/ThreeN/workflows/build/badge.svg)](https://github.com/azixaka/ThreeN/actions)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://github.com/azixaka/ThreeN/blob/master/LICENSE)
[![.NET](https://img.shields.io/badge/.NET-9.0-512BD4)](https://dotnet.microsoft.com/)

A high-performance, zero-dependency neural network library for .NET featuring SIMD vectorization, modern optimizers, and a fluent API for building and training neural networks.

## Features

- **Fluent Builder API** - Intuitive, chainable methods for network construction
- **Multiple Activation Functions** - Sigmoid, ReLU, Tanh, Softmax, Sin, PassThrough
- **Modern Optimizers** - SGD, Adam, Momentum with adaptive learning rates
- **Advanced Training** - L2 regularization, learning rate schedules, early stopping, validation splits
- **Weight Initialization** - Xavier (Glorot), He, and uniform random initialization
- **Loss Functions** - Mean Squared Error, Cross-Entropy
- **Performance Metrics** - Classification accuracy, custom cost functions
- **SIMD Optimizations** - Vectorized matrix operations using `System.Numerics.Vector<T>`
- **Parallel Training** - Multi-threaded backpropagation for large batches
- **Generic Matrix Library** - Type-safe linear algebra with `Matrix<T>`
- **Zero Dependencies** - Pure .NET implementation, no external packages required

## Installation

```bash
dotnet add package ThreeN
```

Or via NuGet Package Manager:

```bash
Install-Package ThreeN
```

## Quick Start

### XOR Network (One Line!)

```csharp
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;

// Prepare XOR dataset
var xorData = new float[] { 0, 0, 0,  0, 1, 1,  1, 0, 1,  1, 1, 0 };

// Create input and output matrices using clear static factories
var inputs = Matrix<float>.FromArrayStrided(4, 2, xorData, startIndex: 0, stride: 3);
var outputs = Matrix<float>.FromArrayStrided(4, 1, xorData, startIndex: 2, stride: 3);

// Build network: 2 inputs → 2 hidden (Sigmoid) → 1 output (Sigmoid)
var network = new NeuralNetworkBuilder()
    .WithInputs(2)
    .WithHiddenLayer(2, Activation.Sigmoid)
    .WithOutputLayer(1, Activation.Sigmoid)
    .WithInitialization(WeightInitialization.Xavier)
    .Build();

// Train with one line! No need to manage gradients manually
network.Train(inputs, outputs, epochs: 100_000, learningRate: 1f);

// Test predictions
for (int i = 0; i < inputs.Rows; i++)
{
    inputs.CopyRow(network.InputLayer, i);
    network.Forward();
    Console.WriteLine($"{inputs[i, 0]} XOR {inputs[i, 1]} = {network.OutputLayer[0, 0]:F4}");
}
// Output: 0 XOR 0 = 0.0056
//         0 XOR 1 = 0.9953
//         1 XOR 0 = 0.9953
//         1 XOR 1 = 0.0048
```

## Network Architecture

### Fluent Builder API

The builder pattern provides a clear, type-safe way to construct networks:

```csharp
var network = new NeuralNetworkBuilder()
    .WithInputs(784)                            // MNIST: 28×28 pixels
    .WithHiddenLayer(128, Activation.Relu)      // Hidden layer
    .WithHiddenLayer(64, Activation.Relu)       // Another hidden layer
    .WithOutputLayer(10, Activation.Softmax)    // 10 classes
    .WithInitialization(WeightInitialization.He) // He init for ReLU
    .Build();
```

### Activation Functions

| Function | Formula | Range | Best For |
|----------|---------|-------|----------|
| **Sigmoid** | σ(x) = 1/(1 + e^(-x)) | (0, 1) | Binary classification output |
| **ReLU** | f(x) = max(0.01x, x) | [0, ∞) | Hidden layers (Leaky ReLU variant) |
| **Tanh** | tanh(x) | (-1, 1) | Hidden layers, zero-centered |
| **Softmax** | exp(x_i) / Σexp(x_j) | (0, 1), Σ=1 | Multi-class classification output |
| **PassThrough** | f(x) = x | (-∞, ∞) | Regression output layers |
| **Sin** | sin(x) | [-1, 1] | Experimental/periodic functions |

### Weight Initialization

```csharp
// He initialization - optimal for ReLU activations
.WithInitialization(WeightInitialization.He)

// Xavier/Glorot initialization - optimal for Sigmoid/Tanh
.WithInitialization(WeightInitialization.Xavier)

// Uniform random in [-1, 1]
.WithInitialization(WeightInitialization.Random)
```

## Training

### Simple Training (Recommended for Beginners)

```csharp
var network = new NeuralNetworkBuilder()
    .WithInputs(2)
    .WithHiddenLayer(5, Activation.Relu)
    .WithOutputLayer(1, Activation.PassThrough)
    .WithInitialization(WeightInitialization.He)
    .Build();

// Simplest: Just train!
network.Train(trainInputs, trainOutputs, epochs: 10_000, learningRate: 0.001f);

// With progress logging
network.Train(trainInputs, trainOutputs, epochs: 10_000, learningRate: 0.001f,
    onEpochComplete: (epoch, loss) =>
    {
        if (epoch % 1000 == 0)
            Console.WriteLine($"Epoch {epoch}: Loss = {loss:F6}");
    });
```

### Training with Modern Optimizers

```csharp
using ThreeN.NeuralNetwork.Optimizers;

var network = new NeuralNetworkBuilder()
    .WithInputs(784)
    .WithHiddenLayer(128, Activation.Relu)
    .WithOutputLayer(10, Activation.Softmax)
    .WithInitialization(WeightInitialization.He)
    .Build();

// Use Adam optimizer for faster convergence
var optimizer = new AdamOptimizer(learningRate: 0.001f);

network.Train(trainInputs, trainOutputs, epochs: 100, optimizer: optimizer,
    onEpochComplete: (epoch, loss) =>
    {
        if (epoch % 10 == 0)
        {
            float accuracy = network.ComputeAccuracy(testInputs, testOutputs);
            Console.WriteLine($"Epoch {epoch}: Accuracy = {accuracy:P2}");
        }
    });
```

### Custom Training Loop (Advanced)

For maximum control, you can still use manual training:

```csharp
// Option 1: TrainStep for simple loops
for (int epoch = 0; epoch < 10_000; epoch++)
{
    network.TrainStep(inputs, outputs, learningRate: 0.001f);

    if (epoch % 1000 == 0)
        Console.WriteLine($"Loss: {network.ComputeCost(inputs, outputs):F6}");
}

// Option 2: Manual gradient management (expert users)
var gradient = Gradient.CreateFor(network);
for (int epoch = 0; epoch < 10_000; epoch++)
{
    NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
    // Custom gradient manipulation here...
    network.ApplyGradient(gradient, learningRate: 0.001f);
}
```

## Optimizers

ThreeN includes modern optimization algorithms with adaptive learning rates:

### SGD (Stochastic Gradient Descent)

```csharp
var optimizer = new SGDOptimizer(learningRate: 0.01f);
```

Classic gradient descent with fixed learning rate.

### Momentum

```csharp
var optimizer = new MomentumOptimizer(learningRate: 0.01f, momentum: 0.9f);
```

Accelerates convergence using exponentially weighted moving average of gradients.

### Adam (Adaptive Moment Estimation)

```csharp
var optimizer = new AdamOptimizer(learningRate: 0.001f)
{
    Beta1 = 0.9f,   // First moment decay
    Beta2 = 0.999f, // Second moment decay
    Epsilon = 1e-8f // Numerical stability
};
```

State-of-the-art optimizer with adaptive learning rates per parameter. Often converges faster than SGD.

## Advanced Training Features

### Trainer Class with Full Features

```csharp
using ThreeN.NeuralNetwork;
using ThreeN.NeuralNetwork.Optimizers;

var network = new NeuralNetworkBuilder()
    .WithInputs(784)
    .WithHiddenLayer(128, Activation.Relu)
    .WithOutputLayer(10, Activation.Softmax)
    .WithInitialization(WeightInitialization.He)
    .Build();

// Create trainer with Adam optimizer and step decay schedule
var trainer = new Trainer(
    network,
    new AdamOptimizer(learningRate: 0.001f),
    new StepDecaySchedule(initialRate: 0.001f, decayFactor: 0.5f, stepSize: 10)
)
{
    L2Lambda = 0.01f,              // L2 regularization
    EarlyStoppingPatience = 20     // Stop if no improvement for 20 epochs
};

// Train with validation split
trainer.Train(
    trainInputs, trainOutputs,
    valInputs, valOutputs,
    epochs: 100
);

// Access training history
Console.WriteLine($"Final train loss: {trainer.History.TrainLoss.Last():F6}");
Console.WriteLine($"Final val loss: {trainer.History.ValLoss.Last():F6}");
Console.WriteLine($"Final val accuracy: {trainer.History.ValAccuracy.Last():P2}");
```

### L2 Regularization

Prevents overfitting by penalizing large weights:

```csharp
// Manual approach
NeuralNetworkExtensions.BackPropagation(
    network, gradient,
    trainInputs, trainOutputs,
    l2Lambda: 0.01f  // Regularization strength
);

float cost = network.ComputeCost(trainInputs, trainOutputs, l2Lambda: 0.01f);
```

### Learning Rate Schedules

Control learning rate decay over time:

```csharp
// Constant learning rate (no decay)
var schedule = new ConstantSchedule(learningRate: 0.001f);

// Step decay: multiply by factor every N epochs
var schedule = new StepDecaySchedule(
    initialRate: 0.01f,
    decayFactor: 0.5f,  // Halve learning rate
    stepSize: 10        // Every 10 epochs
);

// Exponential decay: lr = initial * e^(-decay * epoch)
var schedule = new ExponentialDecaySchedule(
    initialRate: 0.01f,
    decayRate: 0.01f
);
```

### Early Stopping

Automatically stop training when validation loss stops improving:

```csharp
var trainer = new Trainer(network, optimizer)
{
    EarlyStoppingPatience = 20  // Stop after 20 epochs without improvement
};

trainer.Train(trainInputs, trainOutputs, valInputs, valOutputs, epochs: 1000);
// May stop before 1000 epochs if validation loss plateaus
```

## Loss Functions and Metrics

### Mean Squared Error (Regression)

```csharp
float cost = network.ComputeCost(inputs, outputs);
```

### Cross-Entropy Loss (Classification)

```csharp
float loss = NeuralNetworkExtensions.CrossEntropyLoss(network, inputs, outputs);
```

Use with Softmax output layer for multi-class classification.

### Classification Accuracy

```csharp
float accuracy = network.ComputeAccuracy(testInputs, testOutputs);
Console.WriteLine($"Test Accuracy: {accuracy:P2}");  // e.g., "95.23%"
```

Computes the proportion of correctly classified samples (using argmax).

## Linear Algebra

ThreeN includes a generic, type-safe matrix library with SIMD optimizations:

### Matrix Creation and Operations

```csharp
using ThreeN.LinearAlgebra;

// Create a 3×2 matrix
var matrix = new Matrix<float>(rows: 3, columns: 2);

// From existing data
var data = new float[] { 1, 2, 3, 4, 5, 6 };
var matrix = new Matrix<float>(2, 3, data);  // 2 rows × 3 columns

// Matrix operations
MatrixExtensions.Fill(ref matrix, 0f);              // Fill with constant
MatrixExtensions.Randomise(ref matrix, -1f, 1f);    // Uniform random
MatrixExtensions.DotProduct(result, a, b);          // Matrix multiplication
matrix.Add(ref other);                              // Element-wise addition

// Access elements
float value = matrix[row, col];
matrix[row, col] = 3.14f;
```

### Strided Matrix Views

Create matrix views without copying data:

```csharp
var rawData = new float[]
{
    1.0f, 2.0f, 3.0f,  // row 0: input features | output
    4.0f, 5.0f, 6.0f   // row 1: input features | output
};

// View first 2 columns as input matrix
var inputs = new Matrix<float>(
    rows: 2,
    columns: 2,
    startIndex: 0,
    stride: 3,  // Jump by 3 to next row
    data: rawData
);

// View last column as output matrix
var outputs = new Matrix<float>(
    rows: 2,
    columns: 1,
    startIndex: 2,  // Start at column 2
    stride: 3,
    data: rawData
);
```

## Performance

ThreeN is optimized for high performance using SIMD vectorization and parallel processing:

### Optimization Techniques

- **SIMD Vectorization**: Matrix operations use `System.Numerics.Vector<float>` for parallel computation
- **Parallel Backpropagation**: Multi-threaded training for batches ≥100 samples
- **Zero Allocations**: Hot paths (Forward, BackPropagation) allocate no heap memory after initialization
- **Cache-Friendly**: Sequential memory access patterns for optimal CPU cache utilization

### Benchmark Results

Benchmarks run on AMD Ryzen 9 5950X (16 cores), .NET 9.0.10, Windows 11 with AVX2 SIMD support:

#### Matrix Operations

| Operation | Mean Time | Notes |
|-----------|-----------|-------|
| DotProduct 10×10 (SIMD) | 9.4 μs | Small matrix multiplication |
| DotProduct 100×100 (SIMD) | 1.7 ms | Medium matrix multiplication |
| DotProduct 512×512 (Cache-Blocked) | 31.8 ms | Large matrix with cache optimization |
| **DotProduct 784×784 (Cache-Blocked)** | **107.3 ms** | MNIST-sized matrices |
| DotProduct 1000×1000 (Cache-Blocked) | 218.7 ms | Very large matrix multiplication |
| MatrixAdd 784×784 | 112.5 μs | Element-wise addition with SIMD |
| Fill 784×784 | 50.9 μs | Memory fill with SIMD |
| Activate ReLU 784×10 | 86.2 μs | Leaky ReLU activation |
| Activate Softmax 784×10 | 71.1 μs | Softmax with numerical stability |

#### Neural Network Operations

| Operation | Mean Time | Memory | Notes |
|-----------|-----------|--------|-------|
| Forward (10→5→2) | 1.7 μs | 112 B | Small network forward pass |
| Forward (100→50→10) | 20.9 μs | 400 B | Medium network forward pass |
| **Forward (784→128→10)** | **51.7 μs** | **112 B** | **MNIST-sized network** |
| BackProp 50 samples (10→5→2) | 113 μs | 400 B | Small network training |
| BackProp 100 samples (100→50→10) | 1.5 ms | 1.5 MB | Medium network training |
| **BackProp 100 samples (784→128→10)** | **20.4 ms** | **26.4 MB** | **MNIST training (parallel)** |

#### Optimizer Performance (100 epochs on XOR problem)

| Optimizer | Single Update | 100 Epochs | Memory | Ratio vs SGD |
|-----------|--------------|------------|--------|--------------|
| **SGD** | 10.9 μs | 861 μs | 400 B | 1.00× (baseline) |
| **Momentum** | 10.9 μs | 919 μs | 712 B | 1.07× |
| **Adam** | 12.0 μs | 968 μs | 1024 B | 1.12× |

**Key Performance Features:**
- **Cache-Blocked DotProduct**: 7.7× faster than naive implementation for large matrices
- **SIMD Vectorization**: All matrix operations use AVX2 for 4-8× speedup
- **Parallel BackProp**: Automatically parallelizes for batches ≥100 samples
- **Zero Heap Allocations**: Forward and backward passes allocate no memory after initialization

### Running Benchmarks

To reproduce these results:

```bash
cd ThreeN.Benchmarks

# Quick mode (3 warmup + 10 iterations, good balance)
dotnet run -c Release -- --quick

# Standard mode (full benchmark suite, most accurate)
dotnet run -c Release
```

Results are saved to `BenchmarkDotNet.Artifacts/results/` directory.

## Examples

### Linear Regression

```csharp
var trainingData = new float[]
{
    0, 0, 0,   // x1, x2, y
    0, 1, 1,
    1, 0, 1,
    1, 1, 2,
    2, 2, 4,
    3, 2, 5,
    4, 3, 7,
    5, 5, 10
};

var inputs = new Matrix<float>(8, 2, 0, 3, trainingData);
var outputs = new Matrix<float>(8, 1, 2, 3, trainingData);

var network = new NeuralNetworkBuilder()
    .WithInputs(2)
    .WithHiddenLayer(5, Activation.Relu)
    .WithOutputLayer(1, Activation.PassThrough)  // Linear output
    .WithInitialization(WeightInitialization.He)
    .Build();

var gradient = Gradient.CreateFor(network);

for (int epoch = 0; epoch < 10_000; epoch++)
{
    NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
    network.ApplyGradient(gradient, learningRate: 0.001f);
}

// Make predictions
inputs.CopyRow(network.InputLayer, rowIndex: 0);
network.Forward();
Console.WriteLine($"Prediction: {network.OutputLayer[0, 0]:F2}");
```

### Multi-Class Classification (MNIST)

```csharp
// Load MNIST data (28×28 grayscale images, 10 classes)
var trainInputs = LoadMNISTImages("train-images.idx3-ubyte");    // 60000×784
var trainLabels = LoadMNISTLabels("train-labels.idx1-ubyte");    // 60000×10 (one-hot)
var testInputs = LoadMNISTImages("test-images.idx3-ubyte");      // 10000×784
var testLabels = LoadMNISTLabels("test-labels.idx1-ubyte");      // 10000×10

// Build classifier: 784 inputs → 128 hidden (ReLU) → 10 outputs (Softmax)
var network = new NeuralNetworkBuilder()
    .WithInputs(784)
    .WithHiddenLayer(128, Activation.Relu)
    .WithOutputLayer(10, Activation.Softmax)
    .WithInitialization(WeightInitialization.He)
    .Build();

// Train with Adam optimizer and validation
var trainer = new Trainer(
    network,
    new AdamOptimizer(learningRate: 0.001f),
    new StepDecaySchedule(initialRate: 0.001f, decayFactor: 0.9f, stepSize: 10)
)
{
    L2Lambda = 0.0001f,
    EarlyStoppingPatience = 5
};

trainer.Train(trainInputs, trainLabels, testInputs, testLabels, epochs: 50);

// Evaluate
float testAccuracy = network.ComputeAccuracy(testInputs, testLabels);
float testLoss = NeuralNetworkExtensions.CrossEntropyLoss(network, testInputs, testLabels);

Console.WriteLine($"Test Accuracy: {testAccuracy:P2}");
Console.WriteLine($"Test Loss: {testLoss:F4}");
```

### Custom Training Loop with Logging

```csharp
var network = new NeuralNetworkBuilder()
    .WithInputs(2)
    .WithHiddenLayer(10, Activation.Relu)
    .WithOutputLayer(1, Activation.Sigmoid)
    .Build();

var optimizer = new AdamOptimizer(learningRate: 0.001f);
var gradient = Gradient.CreateFor(network);
var schedule = new StepDecaySchedule(initialRate: 0.001f, decayFactor: 0.95f, stepSize: 100);

for (int epoch = 0; epoch < 10_000; epoch++)
{
    // Update learning rate
    optimizer.LearningRate = schedule.GetLearningRate(epoch);

    // Train
    NeuralNetworkExtensions.BackPropagation(network, gradient, trainInputs, trainOutputs, l2Lambda: 0.01f);
    optimizer.Update(network, gradient);

    // Log progress every 500 epochs
    if (epoch % 500 == 0)
    {
        float trainCost = network.ComputeCost(trainInputs, trainOutputs, l2Lambda: 0.01f);
        float valCost = network.ComputeCost(valInputs, valOutputs, l2Lambda: 0.01f);
        float trainAcc = network.ComputeAccuracy(trainInputs, trainOutputs);
        float valAcc = network.ComputeAccuracy(valInputs, valOutputs);

        Console.WriteLine($"Epoch {epoch,5} | LR: {optimizer.LearningRate:F6} | " +
                         $"Train Loss: {trainCost:F4} | Val Loss: {valCost:F4} | " +
                         $"Train Acc: {trainAcc:P1} | Val Acc: {valAcc:P1}");
    }
}
```

## Serialization

Save and load trained networks:

```csharp
using ThreeN.NeuralNetwork;

// Save network to file
NeuralNetworkSerialiser.Save(network, "my_model.nn");

// Load network from file
var loadedNetwork = NeuralNetworkSerialiser.Load("my_model.nn");
```

## Architecture Decisions

### Why Generic Matrix<T>?

ThreeN uses `Matrix<T> where T : INumber<T>` for type safety and potential future support for `double` or `Half` precision:

```csharp
var floatMatrix = new Matrix<float>(10, 10);   // Standard precision
var doubleMatrix = new Matrix<double>(10, 10); // High precision (no SIMD optimizations)
```

SIMD optimizations are currently only enabled for `float` matrices, with automatic fallback to scalar code for other types.

### Zero Dependencies

ThreeN has no external dependencies beyond the .NET Base Class Library. This ensures:

- **Small package size** - No transitive dependencies
- **Easy deployment** - Works anywhere .NET runs
- **Long-term stability** - No breaking changes from third-party packages
- **Full control** - All code is in this library

## Project Structure

```
ThreeN/
├── ThreeN/                          # Core library
│   ├── LinearAlgebra/
│   │   ├── Matrix.cs                # Generic Matrix<T> implementation
│   │   └── MatrixExtensions.cs      # SIMD-optimized operations
│   └── NeuralNetwork/
│       ├── NeuralNetwork.cs         # Network structure
│       ├── NeuralNetworkBuilder.cs  # Fluent builder API
│       ├── Gradient.cs              # Gradient storage
│       ├── ActivationFunctions.cs   # Activation implementations
│       ├── Trainer.cs               # High-level training API
│       ├── TrainingHistory.cs       # Metrics tracking
│       ├── LearningRateSchedule.cs  # LR decay strategies
│       └── Optimizers/
│           ├── IOptimizer.cs
│           ├── SGDOptimizer.cs
│           ├── AdamOptimizer.cs
│           └── MomentumOptimizer.cs
├── NeuralNetwork.Playground/        # XOR and linear regression examples
├── NeuralNetwork.HandWrittenNumbers/ # MNIST classification example
├── ThreeN.Benchmarks/               # Performance benchmarks
└── ThreeN.Tests/                    # Unit tests
```

## Requirements

- **.NET 9.0** or later
- **No external dependencies**

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

ThreeN implements standard neural network algorithms including:

- **Backpropagation**: Rumelhart, Hinton, and Williams (1986)
- **Adam Optimizer**: Kingma and Ba (2014) - [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- **Xavier Initialization**: Glorot and Bengio (2010)
- **He Initialization**: He et al. (2015)
- **L2 Regularization**: Tikhonov regularization / weight decay

## Support

- **Documentation**: This README
- **Issues**: [GitHub Issues](https://github.com/azixaka/ThreeN/issues)
- **Examples**: See `NeuralNetwork.Playground` and `NeuralNetwork.HandWrittenNumbers` projects
