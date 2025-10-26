namespace ThreeN.NeuralNetwork;

/// <summary>
/// Fluent builder for creating and configuring neural networks.
/// </summary>
/// <example>
/// <code>
/// var network = new NeuralNetworkBuilder()
///     .WithInputs(784)
///     .WithHiddenLayer(128, ActivationFunctionType.Relu)
///     .WithOutputLayer(10, ActivationFunctionType.Softmax)
///     .WithInitialization(WeightInitialization.He)
///     .Build();
/// </code>
/// </example>
public sealed class NeuralNetworkBuilder
{
    private readonly List<(int neurons, ActivationFunctionType activation)> _layers = new();
    private int _inputSize;
    private WeightInitialization _initialization = WeightInitialization.He;

    /// <summary>
    /// Specifies the number of input neurons.
    /// Must be called before adding layers.
    /// </summary>
    /// <param name="size">Number of input neurons (must be &gt; 0).</param>
    /// <returns>This builder for method chaining.</returns>
    /// <exception cref="ArgumentException">If size is &lt;= 0.</exception>
    public NeuralNetworkBuilder WithInputs(int size)
    {
        if (size < 1) throw new ArgumentException("Input size must be > 0", nameof(size));
        _inputSize = size;
        return this;
    }

    /// <summary>
    /// Adds a hidden layer.
    /// </summary>
    /// <param name="neurons">Number of neurons in the layer (must be &gt; 0).</param>
    /// <param name="activation">Activation function for the layer.</param>
    /// <returns>This builder for method chaining.</returns>
    /// <exception cref="ArgumentException">If neurons is &lt;= 0.</exception>
    public NeuralNetworkBuilder WithHiddenLayer(int neurons, ActivationFunctionType activation)
    {
        if (neurons < 1) throw new ArgumentException("Layer must have > 0 neurons", nameof(neurons));
        _layers.Add((neurons, activation));
        return this;
    }

    /// <summary>
    /// Adds the output layer.
    /// </summary>
    /// <param name="neurons">Number of output neurons (must be &gt; 0).</param>
    /// <param name="activation">Activation function for the output layer.</param>
    /// <returns>This builder for method chaining.</returns>
    /// <exception cref="ArgumentException">If neurons is &lt;= 0.</exception>
    public NeuralNetworkBuilder WithOutputLayer(int neurons, ActivationFunctionType activation)
    {
        if (neurons < 1) throw new ArgumentException("Output layer must have > 0 neurons", nameof(neurons));
        _layers.Add((neurons, activation));
        return this;
    }

    /// <summary>
    /// Sets the weight initialization method.
    /// Default is He initialization.
    /// </summary>
    /// <param name="method">The initialization method to use.</param>
    /// <returns>This builder for method chaining.</returns>
    public NeuralNetworkBuilder WithInitialization(WeightInitialization method)
    {
        _initialization = method;
        return this;
    }

    /// <summary>
    /// Builds the neural network with the specified configuration.
    /// </summary>
    /// <returns>A configured neural network.</returns>
    /// <exception cref="InvalidOperationException">
    /// If WithInputs() was not called or no layers were added.
    /// </exception>
    public NeuralNetwork Build()
    {
        if (_inputSize == 0)
            throw new InvalidOperationException("Call WithInputs() first to specify input size");

        if (_layers.Count == 0)
            throw new InvalidOperationException("Add at least one layer using WithHiddenLayer() or WithOutputLayer()");

        var config = new int[_layers.Count + 1];
        var activations = new ActivationFunctionType[_layers.Count];

        config[0] = _inputSize;
        for (int i = 0; i < _layers.Count; i++)
        {
            config[i + 1] = _layers[i].neurons;
            activations[i] = _layers[i].activation;
        }

        var nn = NeuralNetwork.Create(activations, config);

        switch (_initialization)
        {
            case WeightInitialization.He:
                nn.InitializeHe();
                break;
            case WeightInitialization.Xavier:
                nn.InitializeXavier();
                break;
            case WeightInitialization.Random:
                nn.Randomize(-1f, 1f);
                break;
        }

        return nn;
    }
}
