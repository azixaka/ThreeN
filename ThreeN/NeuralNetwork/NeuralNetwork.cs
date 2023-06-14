using ThreeN.LinearAlgebra;

namespace ThreeN.NeuralNetwork;

public sealed class NeuralNetwork
{
    public Matrix<float>[] Weights { get; init; }
    public Matrix<float>[] Biases { get; init; }
    public Matrix<float>[] Activations { get; init; }
    public ActivationFunctionType[] ActivationFunctions { get; init; }

    internal NeuralNetwork(Matrix<float>[] weights, Matrix<float>[] biases, Matrix<float>[] activations, ActivationFunctionType[] activationFunctions)
    {
        Weights = weights;
        Biases = biases;
        Activations = activations;
        ActivationFunctions = activationFunctions;
    }

    public Matrix<float> InputLayer => Activations[0];
    public Matrix<float> OutputLayer => Activations[Activations.Length - 1];

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
            activationFunctions[i - 1] = activationFunctions[i - 1];
        }

        return new NeuralNetwork(weights, biases, activations, activationFunctions);
    }
}