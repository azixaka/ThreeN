using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using ThreeN.NeuralNetwork.Optimizers;

BenchmarkRunner.Run<MatrixOperationsBenchmarks>();
BenchmarkRunner.Run<NeuralNetworkBenchmarks>();
BenchmarkRunner.Run<OptimizerBenchmarks>();

[MemoryDiagnoser]
public class MatrixOperationsBenchmarks
{
    private Matrix<float> _a10, _b10, _result10;
    private Matrix<float> _a100, _b100, _result100;
    private Matrix<float> _a784, _b784, _result784;
    private Matrix<float> _addA, _addB;
    private Matrix<float> _fillMatrix;
    private Matrix<float> _activateMatrix;

    [GlobalSetup]
    public void Setup()
    {
        // DotProduct test matrices
        _a10 = new Matrix<float>(10, 10);
        _b10 = new Matrix<float>(10, 10);
        _result10 = new Matrix<float>(10, 10);

        _a100 = new Matrix<float>(100, 100);
        _b100 = new Matrix<float>(100, 100);
        _result100 = new Matrix<float>(100, 100);

        _a784 = new Matrix<float>(784, 784);
        _b784 = new Matrix<float>(784, 784);
        _result784 = new Matrix<float>(784, 784);

        // Add test matrices
        _addA = new Matrix<float>(784, 784);
        _addB = new Matrix<float>(784, 784);

        // Fill test matrix
        _fillMatrix = new Matrix<float>(784, 784);

        // Activate test matrix
        _activateMatrix = new Matrix<float>(784, 10);

        // Initialize with random values
        MatrixExtensions.Randomise(ref _a10, -1f, 1f);
        MatrixExtensions.Randomise(ref _b10, -1f, 1f);
        MatrixExtensions.Randomise(ref _a100, -1f, 1f);
        MatrixExtensions.Randomise(ref _b100, -1f, 1f);
        MatrixExtensions.Randomise(ref _a784, -1f, 1f);
        MatrixExtensions.Randomise(ref _b784, -1f, 1f);
        MatrixExtensions.Randomise(ref _addA, -1f, 1f);
        MatrixExtensions.Randomise(ref _addB, -1f, 1f);
        MatrixExtensions.Randomise(ref _activateMatrix, -1f, 1f);
    }

    [Benchmark]
    public void DotProduct_10x10_Baseline()
    {
        MatrixExtensions.DotProduct(_result10, _a10, _b10);
    }

    [Benchmark]
    public void DotProduct_100x100_Baseline()
    {
        MatrixExtensions.DotProduct(_result100, _a100, _b100);
    }

    [Benchmark]
    public void DotProduct_784x784_Baseline()
    {
        MatrixExtensions.DotProduct(_result784, _a784, _b784);
    }

    [Benchmark]
    public void MatrixAdd_784x784_Baseline()
    {
        _addA.Add(ref _addB);
    }

    [Benchmark]
    public void Fill_784x784_Baseline()
    {
        MatrixExtensions.Fill(ref _fillMatrix, 0.5f);
    }

    [Benchmark]
    public void Activate_ReLU_784x10_Baseline()
    {
        MatrixExtensions.Activate(ref _activateMatrix, Activation.Relu);
    }

    [Benchmark]
    public void Activate_Softmax_784x10_Baseline()
    {
        MatrixExtensions.Activate(ref _activateMatrix, Activation.Softmax);
    }
}

[MemoryDiagnoser]
public class NeuralNetworkBenchmarks
{
    private NeuralNetwork _nn_small = null!;
    private NeuralNetwork _nn_medium = null!;
    private NeuralNetwork _nn_mnist = null!;
    private Gradient _gradient_small = null!;
    private Gradient _gradient_medium = null!;
    private Gradient _gradient_mnist = null!;
    private Matrix<float> _inputs_small;
    private Matrix<float> _outputs_small;
    private Matrix<float> _inputs_medium;
    private Matrix<float> _outputs_medium;
    private Matrix<float> _inputs_mnist;
    private Matrix<float> _outputs_mnist;

    [GlobalSetup]
    public void Setup()
    {
        // Small network: 10-5-2 (for XOR-like problems)
        _nn_small = NeuralNetwork.Create(
            new[] { Activation.Relu, Activation.PassThrough },
            new[] { 10, 5, 2 }
        );
        _gradient_small = Gradient.CreateFor(_nn_small);
        _inputs_small = new Matrix<float>(50, 10);
        _outputs_small = new Matrix<float>(50, 2);

        // Medium network: 100-50-10
        _nn_medium = NeuralNetwork.Create(
            new[] { Activation.Relu, Activation.PassThrough },
            new[] { 100, 50, 10 }
        );
        _gradient_medium = Gradient.CreateFor(_nn_medium);
        _inputs_medium = new Matrix<float>(100, 100);
        _outputs_medium = new Matrix<float>(100, 10);

        // MNIST-like network: 784-128-10
        _nn_mnist = NeuralNetwork.Create(
            new[] { Activation.Relu, Activation.Softmax },
            new[] { 784, 128, 10 }
        );
        _gradient_mnist = Gradient.CreateFor(_nn_mnist);
        _inputs_mnist = new Matrix<float>(100, 784);
        _outputs_mnist = new Matrix<float>(100, 10);

        // Initialize networks and data
        _nn_small.InitializeXavier();
        _nn_medium.InitializeXavier();
        _nn_mnist.InitializeXavier();

        MatrixExtensions.Randomise(ref _inputs_small, -1f, 1f);
        MatrixExtensions.Randomise(ref _outputs_small, -1f, 1f);
        MatrixExtensions.Randomise(ref _inputs_medium, -1f, 1f);
        MatrixExtensions.Randomise(ref _outputs_medium, -1f, 1f);
        MatrixExtensions.Randomise(ref _inputs_mnist, -1f, 1f);
        MatrixExtensions.Randomise(ref _outputs_mnist, -1f, 1f);
    }

    [Benchmark]
    public void Forward_Small_10_5_2_Baseline()
    {
        _inputs_small.CopyRow(_nn_small.InputLayer, 0);
        _nn_small.Forward();
    }

    [Benchmark]
    public void Forward_Medium_100_50_10_Baseline()
    {
        _inputs_medium.CopyRow(_nn_medium.InputLayer, 0);
        _nn_medium.Forward();
    }

    [Benchmark]
    public void Forward_MNIST_784_128_10_Baseline()
    {
        _inputs_mnist.CopyRow(_nn_mnist.InputLayer, 0);
        _nn_mnist.Forward();
    }

    [Benchmark]
    public void BackPropagation_Small_50samples_Baseline()
    {
        NeuralNetworkExtensions.BackPropagation(_nn_small, _gradient_small, _inputs_small, _outputs_small);
    }

    [Benchmark]
    public void BackPropagation_Medium_100samples_Baseline()
    {
        NeuralNetworkExtensions.BackPropagation(_nn_medium, _gradient_medium, _inputs_medium, _outputs_medium);
    }

    [Benchmark]
    public void BackPropagation_MNIST_100samples_Baseline()
    {
        NeuralNetworkExtensions.BackPropagation(_nn_mnist, _gradient_mnist, _inputs_mnist, _outputs_mnist);
    }
}

[MemoryDiagnoser]
public class OptimizerBenchmarks
{
    private NeuralNetwork _nn = null!;
    private Gradient _gradient = null!;
    private Matrix<float> _inputs;
    private Matrix<float> _outputs;
    private SGDOptimizer _sgd = null!;
    private MomentumOptimizer _momentum = null!;
    private AdamOptimizer _adam = null!;

    [GlobalSetup]
    public void Setup()
    {
        // XOR problem
        var rawData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        _inputs = new Matrix<float>(4, 2, 0, 3, rawData);
        _outputs = new Matrix<float>(4, 1, 2, 3, rawData);

        _nn = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(4, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        _gradient = Gradient.CreateFor(_nn);

        // Create optimizers
        _sgd = new SGDOptimizer(learningRate: 0.1f);
        _momentum = new MomentumOptimizer(learningRate: 0.1f, momentum: 0.9f);
        _adam = new AdamOptimizer(learningRate: 0.01f);
    }

    [Benchmark(Baseline = true)]
    public void SGD_SingleUpdate()
    {
        // Reset network for fair comparison
        _nn.InitializeXavier();
        _sgd.Reset();

        // Compute gradients and update
        NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
        _sgd.Update(_nn, _gradient);
    }

    [Benchmark]
    public void Momentum_SingleUpdate()
    {
        // Reset network for fair comparison
        _nn.InitializeXavier();
        _momentum.Reset();

        // Compute gradients and update
        NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
        _momentum.Update(_nn, _gradient);
    }

    [Benchmark]
    public void Adam_SingleUpdate()
    {
        // Reset network for fair comparison
        _nn.InitializeXavier();
        _adam.Reset();

        // Compute gradients and update
        NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
        _adam.Update(_nn, _gradient);
    }

    [Benchmark]
    public void SGD_ConvergenceSpeed()
    {
        // Reset network
        _nn.InitializeXavier();
        _sgd.Reset();
        _sgd.LearningRate = 1f;

        // Train for 100 epochs
        for (int i = 0; i < 100; i++)
        {
            NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
            _sgd.Update(_nn, _gradient);
        }
    }

    [Benchmark]
    public void Momentum_ConvergenceSpeed()
    {
        // Reset network
        _nn.InitializeXavier();
        _momentum.Reset();
        _momentum.LearningRate = 1f;

        // Train for 100 epochs
        for (int i = 0; i < 100; i++)
        {
            NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
            _momentum.Update(_nn, _gradient);
        }
    }

    [Benchmark]
    public void Adam_ConvergenceSpeed()
    {
        // Reset network
        _nn.InitializeXavier();
        _adam.Reset();
        _adam.LearningRate = 0.01f;

        // Train for 100 epochs
        for (int i = 0; i < 100; i++)
        {
            NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
            _adam.Update(_nn, _gradient);
        }
    }
}
