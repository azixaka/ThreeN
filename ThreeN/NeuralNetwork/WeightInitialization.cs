namespace ThreeN.NeuralNetwork;

/// <summary>Weight initialization methods.</summary>
public enum WeightInitialization
{
    /// <summary>He initialization (good for ReLU activations).</summary>
    He,

    /// <summary>Xavier initialization (good for Sigmoid/Tanh activations).</summary>
    Xavier,

    /// <summary>Uniform random initialization in [-1, 1].</summary>
    Random
}
