namespace ThreeN.NeuralNetwork;

/// <summary>
/// Provides activation functions and their derivatives for neural networks.
/// </summary>
/// <remarks>
/// All derivative functions take the OUTPUT (y) of the activation function, not the input (x).
/// This allows efficient backpropagation without recomputing activations.
/// Supported functions: Sigmoid, ReLU (leaky), Tanh, Sin, PassThrough (linear), Softmax.
/// </remarks>
public static class ActivationFunctions
{
    private static readonly float ReluParam = 0.01f;

    /// <summary>
    /// Applies the specified activation function to a value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="type">The activation function type to apply.</param>
    /// <returns>The activated value.</returns>
    /// <exception cref="ArgumentException">If activation function type is invalid or Softmax (use <see cref="ActivateSoftmax"/> for Softmax).</exception>
    public static float Activate(float x, Activation type)
        => type switch
        {
            Activation.Sigmoid => Sigmoid(x),
            Activation.Relu => Relu(x),
            Activation.Tanh => Tanh(x),
            Activation.Sin => Sin(x),
            Activation.PassThrough => PassThrough(x),
            _ => throw new ArgumentException("Invalid activation function type", nameof(type))
        };

    /// <summary>
    /// Computes the derivative of the specified activation function.
    /// </summary>
    /// <param name="y">The OUTPUT of the activation function (not the input x).</param>
    /// <param name="type">The activation function type.</param>
    /// <returns>The derivative value dy/dx evaluated at the output y.</returns>
    /// <exception cref="ArgumentException">If activation function type is invalid or Softmax (Softmax derivative handled differently).</exception>
    /// <remarks>
    /// Takes output y instead of input x for efficiency during backpropagation.
    /// For Sigmoid: derivative = y × (1 - y) where y = sigmoid(x).
    /// </remarks>
    public static float Derivative(float y, Activation type)
        => type switch
        {
            Activation.Sigmoid => SigmoidDerivative(y),
            Activation.Relu => ReluDerivative(y),
            Activation.Tanh => TanhDerivative(y),
            Activation.Sin => SinDerivative(y),
            Activation.PassThrough => PassThroughDerivative(y),
            _ => throw new ArgumentException("Invalid activation function type", nameof(type))
        };

    /// <summary>
    /// Sigmoid activation function: σ(x) = 1 / (1 + e^(-x)).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>A value in range (0, 1).</returns>
    /// <remarks>
    /// Smooth S-shaped curve, good for binary classification.
    /// Output range: (0, 1). Can suffer from vanishing gradients for large |x|.
    /// </remarks>
    public static float Sigmoid(float x)
        => 1f / (1f + (float)Math.Exp(-x));

    /// <summary>
    /// Derivative of Sigmoid: σ'(x) = σ(x) × (1 - σ(x)) = y × (1 - y).
    /// </summary>
    /// <param name="y">The OUTPUT of Sigmoid (not input x).</param>
    /// <returns>The derivative value.</returns>
    public static float SigmoidDerivative(float y)
        => y * (1f - y);

    /// <summary>
    /// Leaky ReLU activation: f(x) = x if x > 0, else 0.01x.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The activated value.</returns>
    /// <remarks>
    /// Leaky ReLU prevents "dying ReLU" problem by allowing small negative gradients.
    /// Leak parameter = 0.01. For standard ReLU, leak = 0.
    /// </remarks>
    public static float Relu(float x)
        => x > 0 ? x : x * ReluParam;

    /// <summary>
    /// Derivative of Leaky ReLU: f'(x) = 1 if x > 0, else 0.01.
    /// </summary>
    /// <param name="y">The OUTPUT of Relu.</param>
    /// <returns>1 if y >= 0, else 0.01.</returns>
    public static float ReluDerivative(float y) =>
        y >= 0 ? 1 : ReluParam;

    /// <summary>
    /// Hyperbolic tangent activation: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)).
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>A value in range (-1, 1).</returns>
    /// <remarks>
    /// Zero-centered output (unlike Sigmoid), often converges faster.
    /// Output range: (-1, 1). Can suffer from vanishing gradients.
    /// </remarks>
    public static float Tanh(float x)
    {
        var exp = (float)Math.Exp(x);
        var negExp = (float)Math.Exp(-x);
        return (exp - negExp) / (exp + negExp);
    }

    /// <summary>
    /// Derivative of Tanh: tanh'(x) = 1 - tanh²(x) = 1 - y².
    /// </summary>
    /// <param name="y">The OUTPUT of Tanh (not input x).</param>
    /// <returns>The derivative value.</returns>
    public static float TanhDerivative(float y)
        => 1f - y * y;

    /// <summary>
    /// Sine activation function: f(x) = sin(x).
    /// </summary>
    /// <param name="x">The input value (in radians).</param>
    /// <returns>A value in range [-1, 1].</returns>
    /// <remarks>
    /// Periodic activation - experimental, not commonly used.
    /// Output range: [-1, 1].
    /// </remarks>
    public static float Sin(float x) =>
        (float)Math.Sin(x);

    /// <summary>
    /// Derivative of Sin: sin'(x) = cos(x) = sqrt(1 - sin²(x)) = sqrt(1 - y²).
    /// </summary>
    /// <param name="y">The OUTPUT of Sin (not input x).</param>
    /// <returns>The derivative value.</returns>
    /// <remarks>
    /// Computes cos(x) from sin(x) = y using identity: cos²(x) + sin²(x) = 1.
    /// Clamps y to [-1, 1] to prevent NaN from numerical errors.
    /// </remarks>
    public static float SinDerivative(float y)
    {
        // Clamp to prevent NaN from numerical errors
        y = Math.Clamp(y, -1f, 1f);
        // cos(x) = sqrt(1 - sin²(x))
        return (float)Math.Sqrt(1f - y * y);
    }

    /// <summary>
    /// PassThrough (linear/identity) activation: f(x) = x.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The input value unchanged.</returns>
    /// <remarks>
    /// No transformation - commonly used for regression output layers.
    /// </remarks>
    public static float PassThrough(float x) => x;

    /// <summary>
    /// Derivative of PassThrough: f'(x) = 1 (constant).
    /// </summary>
    /// <param name="y">The OUTPUT (unused, always returns 1).</param>
    /// <returns>1.</returns>
    public static float PassThroughDerivative(float y) => 1f;

    /// <summary>
    /// Softmax activation function for probability distributions.
    /// </summary>
    /// <param name="values">The span of values to apply Softmax to (modified in-place).</param>
    /// <remarks>
    /// Softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max) where max subtraction provides numerical stability.
    /// Converts a vector of real values into a probability distribution summing to 1.
    /// Output range: (0, 1) for each element, sum = 1.
    /// Typically used for multi-class classification in output layer.
    /// </remarks>
    public static void ActivateSoftmax(Span<float> values)
    {
        if (values.Length == 0) return;

        // Find max for numerical stability
        float max = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > max) max = values[i];

        // Compute exp(x - max) and sum
        float sum = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)Math.Exp(values[i] - max);
            sum += values[i];
        }

        // Normalize
        for (int i = 0; i < values.Length; i++)
            values[i] /= sum;
    }
}
