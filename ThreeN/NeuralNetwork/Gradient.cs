using ThreeN.LinearAlgebra;

namespace ThreeN.NeuralNetwork;

/// <summary>
/// Stores gradients for weights and biases during backpropagation.
/// </summary>
public sealed class Gradient
{
    /// <summary>Weight gradients for each layer.</summary>
    public Matrix<float>[] Weights { get; init; }

    /// <summary>Bias gradients for each layer.</summary>
    public Matrix<float>[] Biases { get; init; }

    /// <summary>Activation gradients for each layer (intermediate values).</summary>
    internal Matrix<float>[] Activations { get; init; }

    internal Gradient(Matrix<float>[] weights, Matrix<float>[] biases, Matrix<float>[] activations)
    {
        Weights = weights;
        Biases = biases;
        Activations = activations;
    }

    /// <summary>
    /// Creates a gradient structure matching the given neural network architecture.
    /// </summary>
    public static Gradient CreateFor(NeuralNetwork network)
    {
        var weights = new Matrix<float>[network.Weights.Length];
        var biases = new Matrix<float>[network.Biases.Length];
        var activations = new Matrix<float>[network.Activations.Length];

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = new Matrix<float>(network.Weights[i].Rows, network.Weights[i].Columns);
            biases[i] = new Matrix<float>(network.Biases[i].Rows, network.Biases[i].Columns);
        }

        for (int i = 0; i < activations.Length; i++)
            activations[i] = new Matrix<float>(network.Activations[i].Rows, network.Activations[i].Columns);

        return new Gradient(weights, biases, activations);
    }

    /// <summary>Fills all gradient values with specified value (typically 0).</summary>
    public void Fill(float value)
    {
        for (int i = 0; i < Weights.Length; i++)
        {
            MatrixExtensions.Fill(ref Weights[i], value);
            MatrixExtensions.Fill(ref Biases[i], value);
            MatrixExtensions.Fill(ref Activations[i], value);
        }
    }
}
