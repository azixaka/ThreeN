namespace ThreeN.NeuralNetwork;

/// <summary>
/// Specifies the activation function to apply in a neural network layer.
/// </summary>
public enum Activation : byte
{
    /// <summary>
    /// Sigmoid activation: σ(x) = 1 / (1 + e^(-x)). Output range: (0, 1).
    /// Good for binary classification, can suffer from vanishing gradients.
    /// </summary>
    Sigmoid,

    /// <summary>
    /// Leaky ReLU activation: f(x) = max(0.01x, x). Output range: (-∞, ∞).
    /// Best for hidden layers, prevents dying ReLU problem. Use with He initialization.
    /// </summary>
    Relu,

    /// <summary>
    /// Hyperbolic tangent activation: tanh(x). Output range: (-1, 1).
    /// Zero-centered, good for hidden layers. Use with Xavier initialization.
    /// </summary>
    Tanh,

    /// <summary>
    /// Sine activation: sin(x). Output range: [-1, 1].
    /// Periodic activation - experimental, not commonly used in practice.
    /// </summary>
    Sin,

    /// <summary>
    /// PassThrough (linear/identity) activation: f(x) = x. Output range: (-∞, ∞).
    /// No transformation - commonly used for regression output layers.
    /// </summary>
    PassThrough,

    /// <summary>
    /// Softmax activation: exp(x_i) / Σexp(x_j). Output range: (0, 1), sum = 1.
    /// Converts logits to probability distribution. Use for multi-class classification output layer.
    /// </summary>
    Softmax
}
