using ThreeN.LinearAlgebra;

namespace ThreeN.NeuralNetwork.Optimizers;

/// <summary>
/// Adam optimizer (Adaptive Moment Estimation).
/// Reference: https://arxiv.org/abs/1412.6980
/// </summary>
public sealed class AdamOptimizer : IOptimizer
{
    public float LearningRate { get; set; }
    public float Beta1 { get; set; } = 0.9f;   // First moment decay
    public float Beta2 { get; set; } = 0.999f; // Second moment decay
    public float Epsilon { get; set; } = 1e-8f; // Numerical stability

    private Matrix<float>[]? _mWeights, _vWeights; // First/second moments for weights
    private Matrix<float>[]? _mBiases, _vBiases;   // First/second moments for biases
    private int _timestep = 0;

    public AdamOptimizer(float learningRate = 0.001f)
    {
        if (learningRate <= 0) throw new ArgumentException("Learning rate must be > 0");
        LearningRate = learningRate;
    }

    public void Update(NeuralNetwork network, Gradient gradient)
    {
        // Lazy initialization
        if (_mWeights == null)
            InitializeState(network);

        _timestep++;
        float beta1_t = (float)Math.Pow(Beta1, _timestep);
        float beta2_t = (float)Math.Pow(Beta2, _timestep);

        for (int i = 0; i < network.Weights.Length; i++)
        {
            UpdateParameters(network.Weights[i], gradient.Weights[i],
                _mWeights![i], _vWeights![i], beta1_t, beta2_t);

            UpdateParameters(network.Biases[i], gradient.Biases[i],
                _mBiases![i], _vBiases![i], beta1_t, beta2_t);
        }
    }

    private void UpdateParameters(Matrix<float> param, Matrix<float> grad,
        Matrix<float> m, Matrix<float> v, float beta1_t, float beta2_t)
    {
        for (int j = 0; j < param.Rows; j++)
        for (int k = 0; k < param.Columns; k++)
        {
            float g = grad[j, k];

            // Update biased first moment
            m[j, k] = Beta1 * m[j, k] + (1 - Beta1) * g;

            // Update biased second moment
            v[j, k] = Beta2 * v[j, k] + (1 - Beta2) * g * g;

            // Bias correction
            float m_hat = m[j, k] / (1 - beta1_t);
            float v_hat = v[j, k] / (1 - beta2_t);

            // Update parameter
            param[j, k] -= LearningRate * m_hat / ((float)Math.Sqrt(v_hat) + Epsilon);
        }
    }

    private void InitializeState(NeuralNetwork network)
    {
        _mWeights = new Matrix<float>[network.Weights.Length];
        _vWeights = new Matrix<float>[network.Weights.Length];
        _mBiases = new Matrix<float>[network.Biases.Length];
        _vBiases = new Matrix<float>[network.Biases.Length];

        for (int i = 0; i < network.Weights.Length; i++)
        {
            _mWeights[i] = new Matrix<float>(network.Weights[i].Rows, network.Weights[i].Columns);
            _vWeights[i] = new Matrix<float>(network.Weights[i].Rows, network.Weights[i].Columns);
            _mBiases[i] = new Matrix<float>(network.Biases[i].Rows, network.Biases[i].Columns);
            _vBiases[i] = new Matrix<float>(network.Biases[i].Rows, network.Biases[i].Columns);
        }
    }

    public void Reset()
    {
        _mWeights = _vWeights = _mBiases = _vBiases = null;
        _timestep = 0;
    }
}
