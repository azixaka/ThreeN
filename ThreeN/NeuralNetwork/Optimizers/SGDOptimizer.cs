namespace ThreeN.NeuralNetwork.Optimizers;

/// <summary>
/// Stochastic Gradient Descent optimizer.
/// </summary>
public sealed class SGDOptimizer : IOptimizer
{
    public float LearningRate { get; set; }

    public SGDOptimizer(float learningRate = 0.01f)
    {
        if (learningRate <= 0) throw new ArgumentException("Learning rate must be > 0");
        LearningRate = learningRate;
    }

    public void Update(NeuralNetwork network, Gradient gradient)
    {
        for (int i = 0; i < network.Weights.Length; i++)
        {
            for (int j = 0; j < network.Weights[i].Rows; j++)
            for (int k = 0; k < network.Weights[i].Columns; k++)
                network.Weights[i][j, k] -= LearningRate * gradient.Weights[i][j, k];

            for (int j = 0; j < network.Biases[i].Rows; j++)
            for (int k = 0; k < network.Biases[i].Columns; k++)
                network.Biases[i][j, k] -= LearningRate * gradient.Biases[i][j, k];
        }
    }

    public void Reset() { /* No state to reset */ }
}
