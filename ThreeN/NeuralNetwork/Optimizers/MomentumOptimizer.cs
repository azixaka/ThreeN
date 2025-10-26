using ThreeN.LinearAlgebra;

namespace ThreeN.NeuralNetwork.Optimizers;

/// <summary>
/// SGD with momentum optimizer.
/// </summary>
public sealed class MomentumOptimizer : IOptimizer
{
    public float LearningRate { get; set; }
    public float Momentum { get; set; } = 0.9f;

    private Matrix<float>[]? _velocityWeights;
    private Matrix<float>[]? _velocityBiases;

    public MomentumOptimizer(float learningRate = 0.01f, float momentum = 0.9f)
    {
        if (learningRate <= 0) throw new ArgumentException("Learning rate must be > 0");
        if (momentum < 0 || momentum >= 1) throw new ArgumentException("Momentum must be in [0, 1)");

        LearningRate = learningRate;
        Momentum = momentum;
    }

    public void Update(NeuralNetwork network, Gradient gradient)
    {
        if (_velocityWeights == null)
            InitializeState(network);

        for (int i = 0; i < network.Weights.Length; i++)
        {
            UpdateWithMomentum(network.Weights[i], gradient.Weights[i], _velocityWeights![i]);
            UpdateWithMomentum(network.Biases[i], gradient.Biases[i], _velocityBiases![i]);
        }
    }

    private void UpdateWithMomentum(Matrix<float> param, Matrix<float> grad, Matrix<float> velocity)
    {
        for (int j = 0; j < param.Rows; j++)
        for (int k = 0; k < param.Columns; k++)
        {
            velocity[j, k] = Momentum * velocity[j, k] - LearningRate * grad[j, k];
            param[j, k] += velocity[j, k];
        }
    }

    private void InitializeState(NeuralNetwork network)
    {
        _velocityWeights = new Matrix<float>[network.Weights.Length];
        _velocityBiases = new Matrix<float>[network.Biases.Length];

        for (int i = 0; i < network.Weights.Length; i++)
        {
            _velocityWeights[i] = new Matrix<float>(network.Weights[i].Rows, network.Weights[i].Columns);
            _velocityBiases[i] = new Matrix<float>(network.Biases[i].Rows, network.Biases[i].Columns);
        }
    }

    public void Reset()
    {
        _velocityWeights = _velocityBiases = null;
    }
}
