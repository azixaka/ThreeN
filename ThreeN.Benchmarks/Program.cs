using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using ThreeN.NeuralNetwork.Optimizers;

// Check for --quick flag to run faster benchmarks
var isQuickMode = args.Contains("--quick") || args.Contains("-q");

var config = isQuickMode
    ? ManualConfig.Create(DefaultConfig.Instance)
        .AddJob(Job.Default
            .WithWarmupCount(3)      // 3 warmup iterations (vs 1) for stable CPU/cache state
            .WithIterationCount(10)  // 10 measurement iterations (vs 3) for better statistics
            .WithInvocationCount(1)  // 1 invocation per iteration
            .WithUnrollFactor(1))    // Don't unroll
    : DefaultConfig.Instance;

Console.WriteLine(isQuickMode
    ? "Running in QUICK mode (3 warmup + 10 iterations - good balance of speed and accuracy)"
    : "Running in STANDARD mode (full benchmark suite, most accurate)");
Console.WriteLine("Use --quick or -q flag for quick mode\n");

BenchmarkRunner.Run<MatrixOperationsBenchmarks>(config);
BenchmarkRunner.Run<NeuralNetworkBenchmarks>(config);
BenchmarkRunner.Run<OptimizerBenchmarks>(config);

/// <summary>
/// Benchmarks for matrix operations including DotProduct with cache-blocking optimization.
///
/// DotProduct uses adaptive strategy:
/// - Small matrices (<512): SIMD with ijk loop order
/// - Large matrices (≥512): Cache-blocked with ikj loop order + SIMD
///
/// Baseline (before cache-blocking, from initial run):
/// - DotProduct_784x784: ~798ms (ijk order, strided column access)
///
/// Expected improvement with cache-blocking:
/// - 10-30x speedup on large matrices due to sequential memory access
/// </summary>
[MemoryDiagnoser]
public class MatrixOperationsBenchmarks
{
    private Matrix<float> _a10, _b10, _result10;
    private Matrix<float> _a100, _b100, _result100;
    private Matrix<float> _a512, _b512, _result512;
    private Matrix<float> _a784, _b784, _result784;
    private Matrix<float> _a1000, _b1000, _result1000;
    private Matrix<float> _addA, _addB;
    private Matrix<float> _fillMatrix;
    private Matrix<float> _activateMatrix;

    [GlobalSetup]
    public void Setup()
    {
        // DotProduct test matrices - various sizes to test both code paths
        _a10 = new Matrix<float>(10, 10);
        _b10 = new Matrix<float>(10, 10);
        _result10 = new Matrix<float>(10, 10);

        _a100 = new Matrix<float>(100, 100);
        _b100 = new Matrix<float>(100, 100);
        _result100 = new Matrix<float>(100, 100);

        _a512 = new Matrix<float>(512, 512);
        _b512 = new Matrix<float>(512, 512);
        _result512 = new Matrix<float>(512, 512);

        _a784 = new Matrix<float>(784, 784);
        _b784 = new Matrix<float>(784, 784);
        _result784 = new Matrix<float>(784, 784);

        _a1000 = new Matrix<float>(1000, 1000);
        _b1000 = new Matrix<float>(1000, 1000);
        _result1000 = new Matrix<float>(1000, 1000);

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
        MatrixExtensions.Randomise(ref _a512, -1f, 1f);
        MatrixExtensions.Randomise(ref _b512, -1f, 1f);
        MatrixExtensions.Randomise(ref _a784, -1f, 1f);
        MatrixExtensions.Randomise(ref _b784, -1f, 1f);
        MatrixExtensions.Randomise(ref _a1000, -1f, 1f);
        MatrixExtensions.Randomise(ref _b1000, -1f, 1f);
        MatrixExtensions.Randomise(ref _addA, -1f, 1f);
        MatrixExtensions.Randomise(ref _addB, -1f, 1f);
        MatrixExtensions.Randomise(ref _activateMatrix, -1f, 1f);
    }

    [Benchmark]
    public void DotProduct_10x10_SIMD()
    {
        // Uses SIMD_IJK path (small matrix)
        MatrixExtensions.DotProduct(_result10, _a10, _b10);
    }

    [Benchmark]
    public void DotProduct_100x100_SIMD()
    {
        // Uses SIMD_IJK path (medium matrix)
        MatrixExtensions.DotProduct(_result100, _a100, _b100);
    }

    [Benchmark]
    public void DotProduct_512x512_CacheBlocked()
    {
        // At threshold: uses cache-blocked path
        MatrixExtensions.DotProduct(_result512, _a512, _b512);
    }

    [Benchmark]
    public void DotProduct_784x784_CacheBlocked()
    {
        // Uses cache-blocked path (MNIST size)
        // Baseline: ~798ms (before cache-blocking)
        // Expected: ~30-80ms (10-25x speedup)
        MatrixExtensions.DotProduct(_result784, _a784, _b784);
    }

    [Benchmark]
    public void DotProduct_1000x1000_CacheBlocked()
    {
        // Uses cache-blocked path (large matrix)
        MatrixExtensions.DotProduct(_result1000, _a1000, _b1000);
    }

    [Benchmark]
    public void MatrixAdd_784x784()
    {
        _addA.Add(ref _addB);
    }

    [Benchmark]
    public void Fill_784x784()
    {
        MatrixExtensions.Fill(ref _fillMatrix, 0.5f);
    }

    [Benchmark]
    public void Activate_ReLU_784x10()
    {
        MatrixExtensions.Activate(ref _activateMatrix, Activation.Relu);
    }

    [Benchmark]
    public void Activate_Softmax_784x10()
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
    public void Forward_Small_10_5_2()
    {
        _inputs_small.CopyRow(_nn_small.InputLayer, 0);
        _nn_small.Forward();
    }

    [Benchmark]
    public void Forward_Medium_100_50_10()
    {
        _inputs_medium.CopyRow(_nn_medium.InputLayer, 0);
        _nn_medium.Forward();
    }

    [Benchmark]
    public void Forward_MNIST_784_128_10()
    {
        _inputs_mnist.CopyRow(_nn_mnist.InputLayer, 0);
        _nn_mnist.Forward();
    }

    [Benchmark]
    public void BackPropagation_Small_50samples()
    {
        NeuralNetworkExtensions.BackPropagation(_nn_small, _gradient_small, _inputs_small, _outputs_small);
    }

    [Benchmark]
    public void BackPropagation_Medium_100samples()
    {
        NeuralNetworkExtensions.BackPropagation(_nn_medium, _gradient_medium, _inputs_medium, _outputs_medium);
    }

    [Benchmark]
    public void BackPropagation_MNIST_100samples()
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
    public void Optimizer_SGD_SingleUpdate()
    {
        // Reset network for fair comparison
        _nn.InitializeXavier();
        _sgd.Reset();

        // Compute gradients and update
        NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
        _sgd.Update(_nn, _gradient);
    }

    [Benchmark]
    public void Optimizer_Momentum_SingleUpdate()
    {
        // Reset network for fair comparison
        _nn.InitializeXavier();
        _momentum.Reset();

        // Compute gradients and update
        NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
        _momentum.Update(_nn, _gradient);
    }

    [Benchmark]
    public void Optimizer_Adam_SingleUpdate()
    {
        // Reset network for fair comparison
        _nn.InitializeXavier();
        _adam.Reset();

        // Compute gradients and update
        NeuralNetworkExtensions.BackPropagation(_nn, _gradient, _inputs, _outputs);
        _adam.Update(_nn, _gradient);
    }

    [Benchmark]
    public void Optimizer_SGD_Convergence_100epochs()
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
    public void Optimizer_Momentum_Convergence_100epochs()
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
    public void Optimizer_Adam_Convergence_100epochs()
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
