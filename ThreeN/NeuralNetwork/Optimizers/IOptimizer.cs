namespace ThreeN.NeuralNetwork.Optimizers;

/// <summary>
/// Interface for gradient-based optimizers.
/// </summary>
public interface IOptimizer
{
    /// <summary>
    /// Updates network parameters using computed gradients.
    /// </summary>
    /// <param name="network">The neural network to update.</param>
    /// <param name="gradient">The computed gradients.</param>
    void Update(NeuralNetwork network, Gradient gradient);

    /// <summary>Gets or sets the learning rate.</summary>
    float LearningRate { get; set; }

    /// <summary>Resets optimizer state (e.g., momentum buffers).</summary>
    void Reset();
}
