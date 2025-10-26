using ThreeN.LinearAlgebra;

namespace ThreeN.NeuralNetwork;

/// <summary>
/// A fully-connected feedforward neural network with configurable architecture and activation functions.
/// </summary>
/// <remarks>
/// Supports multiple hidden layers with different activation functions per layer.
/// Uses efficient matrix operations with SIMD optimizations for float computations.
/// Network state is stored in Activations arrays for forward propagation.
/// Use <see cref="NeuralNetworkBuilder"/> for fluent construction API.
/// </remarks>
public sealed class NeuralNetwork
{
    /// <summary>Weight matrices for each layer connection.</summary>
    /// <remarks>Weights[i] connects Activations[i] to Activations[i+1].</remarks>
    public Matrix<float>[] Weights { get; init; }

    /// <summary>Bias vectors for each layer.</summary>
    /// <remarks>Biases[i] is added after Weights[i] multiplication.</remarks>
    public Matrix<float>[] Biases { get; init; }

    /// <summary>Activation values for each layer (including input and output layers).</summary>
    /// <remarks>Activations[0] is the input layer, Activations[^1] is the output layer.</remarks>
    public Matrix<float>[] Activations { get; init; }

    /// <summary>Activation function types for each layer transition.</summary>
    /// <remarks>ActivationFunctions[i] is applied after Weights[i] × Activations[i] + Biases[i].</remarks>
    public ActivationFunctionType[] ActivationFunctions { get; init; }

    internal NeuralNetwork(Matrix<float>[] weights, Matrix<float>[] biases, Matrix<float>[] activations, ActivationFunctionType[] activationFunctions)
    {
        Weights = weights;
        Biases = biases;
        Activations = activations;
        ActivationFunctions = activationFunctions;
    }

    /// <summary>Gets the input layer (first activation layer).</summary>
    /// <remarks>
    /// Set input values by copying data into this layer before calling <see cref="Forward"/>.
    /// Use <see cref="Matrix{T}.CopyRow"/> to load individual samples.
    /// </remarks>
    public Matrix<float> InputLayer => Activations[0];

    /// <summary>Gets the output layer (final activation layer).</summary>
    /// <remarks>
    /// Contains network predictions after calling <see cref="Forward"/>.
    /// For classification, use argmax to get predicted class from one-hot output.
    /// </remarks>
    public Matrix<float> OutputLayer => Activations[Activations.Length - 1];

    /// <summary>
    /// Performs forward propagation through the network.
    /// </summary>
    /// <remarks>
    /// Computes: for each layer i, Activations[i+1] = Activate(Activations[i] × Weights[i] + Biases[i]).
    /// Input must be set in <see cref="InputLayer"/> before calling this method.
    /// Results are available in <see cref="OutputLayer"/> after completion.
    /// Zero allocations - all computations use pre-allocated activation matrices.
    /// </remarks>
    public void Forward()
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            MatrixExtensions.DotProduct(Activations[i + 1], Activations[i], Weights[i]);
            Activations[i + 1].Add(ref Biases[i]);
            MatrixExtensions.Activate(ref Activations[i + 1], ActivationFunctions[i]);
        }
    }

    /// <summary>
    /// Computes mean squared error (MSE) cost with optional L2 regularization.
    /// </summary>
    /// <param name="inputs">Input data matrix (rows = samples, columns = features).</param>
    /// <param name="expectedOutputs">Expected output data matrix (rows = samples, columns = outputs).</param>
    /// <param name="l2Lambda">L2 regularization strength (0 = no regularization, typical values: 0.001-0.1).</param>
    /// <returns>The average MSE cost across all samples, plus L2 penalty if enabled.</returns>
    /// <remarks>
    /// MSE = (1/n) × Σ(predicted - expected)².
    /// L2 penalty = (λ/2) × Σ(weights²) helps prevent overfitting.
    /// Runs forward propagation for each sample to compute predictions.
    /// </remarks>
    public float ComputeCost(Matrix<float> inputs, Matrix<float> expectedOutputs, float l2Lambda = 0f)
    {
        float cost = 0f;
        for (int i = 0; i < inputs.Rows; i++)
        {
            inputs.CopyRow(InputLayer, i);
            Forward();

            for (int j = 0; j < expectedOutputs.Columns; j++)
            {
                var d = OutputLayer[0, j] - expectedOutputs[i, j];
                cost += d * d;
            }
        }

        cost /= inputs.Rows;

        // Add L2 penalty
        if (l2Lambda > 0f)
        {
            float l2Penalty = 0f;
            for (int i = 0; i < Weights.Length; i++)
            {
                for (int j = 0; j < Weights[i].Rows; j++)
                for (int k = 0; k < Weights[i].Columns; k++)
                    l2Penalty += Weights[i][j, k] * Weights[i][j, k];
            }
            cost += 0.5f * l2Lambda * l2Penalty;
        }

        return cost;
    }

    /// <summary>
    /// Computes cross-entropy loss for classification tasks.
    /// </summary>
    /// <param name="inputs">Input data matrix (rows = samples, columns = features).</param>
    /// <param name="expectedOutputs">Expected output data matrix (rows = samples, columns = classes, one-hot encoded).</param>
    /// <returns>The average cross-entropy loss across all samples.</returns>
    /// <exception cref="ArgumentException">If input/output dimensions don't match network architecture.</exception>
    /// <remarks>
    /// Cross-entropy = -(1/n) × Σ Σ (y × log(p)) where y is true label, p is predicted probability.
    /// Typically used with Softmax output layer for multi-class classification.
    /// Uses clamping to prevent log(0) = -∞ errors.
    /// </remarks>
    public float ComputeCrossEntropyLoss(Matrix<float> inputs, Matrix<float> expectedOutputs)
    {
        if (inputs.Rows != expectedOutputs.Rows)
            throw new ArgumentException("Number of samples in input and output data must be equal");

        if (expectedOutputs.Columns != OutputLayer.Columns)
            throw new ArgumentException("Number of output neurons must equal number of columns in output data");

        float loss = 0f;

        for (int i = 0; i < inputs.Rows; i++)
        {
            inputs.CopyRow(InputLayer, i);
            Forward();

            for (int j = 0; j < expectedOutputs.Columns; j++)
            {
                float y = expectedOutputs[i, j];
                float p = Math.Clamp(OutputLayer[0, j], 1e-7f, 1f - 1e-7f);
                loss -= y * (float)Math.Log(p);
            }
        }

        return loss / inputs.Rows;
    }

    /// <summary>
    /// Computes classification accuracy for one-hot encoded labels.
    /// </summary>
    /// <param name="inputs">Input data matrix (rows = samples, columns = features).</param>
    /// <param name="expectedOutputs">Expected output data matrix (rows = samples, columns = classes, one-hot encoded).</param>
    /// <returns>Accuracy as a fraction in [0, 1] (e.g., 0.95 = 95% accuracy).</returns>
    /// <exception cref="ArgumentException">If input/output dimensions don't match network architecture.</exception>
    /// <remarks>
    /// Accuracy = (correct predictions) / (total samples).
    /// Predicted class = argmax(network output), True class = argmax(expected output).
    /// Returns 1.0 if all predictions are correct, 0.0 if all are wrong.
    /// </remarks>
    public float ComputeAccuracy(Matrix<float> inputs, Matrix<float> expectedOutputs)
    {
        if (inputs.Rows != expectedOutputs.Rows)
            throw new ArgumentException("Number of samples in input and output data must be equal");

        if (expectedOutputs.Columns != OutputLayer.Columns)
            throw new ArgumentException("Number of output neurons must equal number of columns in output data");

        int correct = 0;

        for (int i = 0; i < inputs.Rows; i++)
        {
            inputs.CopyRow(InputLayer, i);
            Forward();

            int predicted = ArgMax(OutputLayer, row: 0);
            int actual = ArgMax(expectedOutputs, row: i);

            if (predicted == actual) correct++;
        }

        return (float)correct / inputs.Rows;
    }

    private static int ArgMax(Matrix<float> matrix, int row)
    {
        int maxIdx = 0;
        float maxVal = matrix[row, 0];

        for (int i = 1; i < matrix.Columns; i++)
        {
            if (matrix[row, i] > maxVal)
            {
                maxVal = matrix[row, i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /// <summary>
    /// Applies gradient descent weight update: weights -= learning_rate × gradient.
    /// </summary>
    /// <param name="gradient">The gradient computed by backpropagation.</param>
    /// <param name="learningRate">The learning rate (step size, typical values: 0.001-0.1).</param>
    /// <remarks>
    /// Updates both weights and biases using vanilla gradient descent.
    /// For advanced optimizers (Adam, Momentum), use <see cref="Optimizers.IOptimizer"/> instead.
    /// Zero allocations - modifies weights and biases in-place.
    /// </remarks>
    public void ApplyGradient(Gradient gradient, float learningRate)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            for (int j = 0; j < Weights[i].Rows; j++)
                for (int k = 0; k < Weights[i].Columns; k++)
                    Weights[i][j, k] -= learningRate * gradient.Weights[i][j, k];

            for (int j = 0; j < Biases[i].Rows; j++)
                for (int k = 0; k < Biases[i].Columns; k++)
                    Biases[i][j, k] -= learningRate * gradient.Biases[i][j, k];
        }
    }

    /// <summary>
    /// Fills all weights, biases, and activations with a constant value.
    /// </summary>
    /// <param name="value">The value to fill with (commonly 0 for reset).</param>
    /// <remarks>
    /// Useful for resetting network state or initializing to specific values.
    /// For proper weight initialization, use <see cref="InitializeHe"/> or <see cref="InitializeXavier"/> instead.
    /// </remarks>
    public void Fill(float value)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            MatrixExtensions.Fill(ref Weights[i], value);
            MatrixExtensions.Fill(ref Biases[i], value);
            MatrixExtensions.Fill(ref Activations[i], value);
        }

        MatrixExtensions.Fill(ref Activations[Weights.Length], value);
    }

    /// <summary>
    /// Initializes weights using Xavier (Glorot) initialization.
    /// </summary>
    /// <remarks>
    /// Xavier initialization: weights ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out))).
    /// Best for Sigmoid and Tanh activations - maintains variance across layers.
    /// Biases are initialized to zero.
    /// Reference: Glorot & Bengio, 2010 - "Understanding the difficulty of training deep feedforward neural networks"
    /// </remarks>
    public void InitializeXavier()
    {
        var rand = Random.Shared;
        for (int i = 0; i < Weights.Length; i++)
        {
            float fan_in = Weights[i].Rows;
            float fan_out = Weights[i].Columns;
            float limit = (float)Math.Sqrt(6.0 / (fan_in + fan_out));

            for (int j = 0; j < Weights[i].Rows; j++)
                for (int k = 0; k < Weights[i].Columns; k++)
                {
                    Weights[i][j, k] = 2 * limit * (float)rand.NextDouble() - limit;
                }

            MatrixExtensions.Fill(ref Biases[i], 0);
        }
    }

    /// <summary>
    /// Initializes weights using He initialization.
    /// </summary>
    /// <remarks>
    /// He initialization: weights ~ Normal(0, √(2/fan_in)).
    /// Best for ReLU activations - prevents vanishing/exploding gradients.
    /// Biases are initialized to zero.
    /// Reference: He et al., 2015 - "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet"
    /// </remarks>
    public void InitializeHe()
    {
        var rand = Random.Shared;
        for (int i = 0; i < Weights.Length; i++)
        {
            float fan_in = Weights[i].Rows;

            for (int j = 0; j < Weights[i].Rows; j++)
                for (int k = 0; k < Weights[i].Columns; k++)
                {
                    Weights[i][j, k] = (float)(rand.NextDouble() * Math.Sqrt(2.0 / fan_in));
                }

            MatrixExtensions.Fill(ref Biases[i], 0);
        }
    }

    /// <summary>
    /// Initializes weights and biases with random values uniformly distributed in [low, high).
    /// </summary>
    /// <param name="low">The inclusive lower bound of the random range.</param>
    /// <param name="high">The exclusive upper bound of the random range.</param>
    /// <remarks>
    /// Simple uniform random initialization - use for testing or when other methods aren't suitable.
    /// For production use, prefer <see cref="InitializeHe"/> (for ReLU) or <see cref="InitializeXavier"/> (for Sigmoid/Tanh).
    /// </remarks>
    public void Randomize(float low, float high)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            MatrixExtensions.Randomise(ref Weights[i], low, high);
            MatrixExtensions.Randomise(ref Biases[i], low, high);
        }
    }

    /// <summary>
    /// Creates a new neural network with the specified architecture and activation functions.
    /// </summary>
    /// <param name="activationFunctions">Activation functions for each layer transition (length = layers - 1).</param>
    /// <param name="configuration">Number of neurons per layer [input, hidden1, ..., output] (minimum length = 2).</param>
    /// <returns>A new neural network with uninitialized weights (all zeros).</returns>
    /// <exception cref="ArgumentException">If configuration or activationFunctions arrays have invalid lengths.</exception>
    /// <remarks>
    /// Configuration example: [784, 128, 10] creates network with 784 inputs, 128 hidden neurons, 10 outputs.
    /// ActivationFunctions example: [ReLU, Softmax] for the 3-layer network above.
    /// Weights are NOT initialized - call <see cref="InitializeHe"/>, <see cref="InitializeXavier"/>, or <see cref="Randomize"/> after creation.
    /// For fluent API, use <see cref="NeuralNetworkBuilder"/> instead (recommended).
    /// </remarks>
    /// <example>
    /// <code>
    /// var nn = NeuralNetwork.Create(
    ///     new[] { ActivationFunctionType.Relu, ActivationFunctionType.Softmax },
    ///     new[] { 784, 128, 10 }
    /// );
    /// nn.InitializeHe();
    /// </code>
    /// </example>
    public static NeuralNetwork Create(ActivationFunctionType[] activationFunctions, int[] configuration) // {2, 2, 1}
    {
        if (configuration.Length < 2)
            throw new ArgumentException("Minimum number of layers is 2 - one for input and one for neurons");

        if (configuration.Length != activationFunctions.Length + 1)
            throw new ArgumentException("Number of activation functions must be equal to number of layers minus one");

        var count = configuration.Length - 1;

        var weights = new Matrix<float>[count];
        var biases = new Matrix<float>[count];
        var activations = new Matrix<float>[configuration.Length];

        activations[0] = new Matrix<float>(1, configuration[0]);
        for (int i = 1; i < configuration.Length; i++)
        {
            weights[i - 1] = new Matrix<float>(activations[i - 1].Columns, configuration[i]);
            biases[i - 1] = new Matrix<float>(1, configuration[i]);
            activations[i] = new Matrix<float>(1, configuration[i]);
        }

        return new NeuralNetwork(weights, biases, activations, activationFunctions);
    }
}