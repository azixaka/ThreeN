using System.Diagnostics;
using System.Numerics;
using System.Runtime.Versioning;
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;

namespace NeuralNetwork.HandWrittenNumbers;

sealed class Program
{
    static void Main()
    {
        // Quick training test with optimizations
        TrainNewModel();

        // Verify the newly trained model works
        VerifyNewModel();
    }

    private static void TrainNewModel()
    {
        Console.WriteLine("=== Training New MNIST Model (Quick Test) ===\n");

        var mnistReader = new MnistReader();
        var (inData, outData) = mnistReader.LoadData(
            Path.Combine("mnist", "train-images.idx3-ubyte"),
            Path.Combine("mnist", "train-labels.idx1-ubyte"));

        Console.WriteLine($"Loaded training data: {inData.Rows} samples\n");

        // Train from scratch (no base model) for just 3 batches to test optimizations
        Train(inData, outData,
            baseModelPath: null,
            destinationModelPath: Path.Combine("mnist", "new-trained-model.3n"),
            samplesPerBath: 100,
            numberOfBatches: 3); // Just 3 batches for quick test

        Console.WriteLine("\n=== Training Complete ===\n");
    }

    private static void VerifyNewModel()
    {
        Console.WriteLine("=== Verifying New Trained Model ===\n");

        var mnistReader = new MnistReader();
        var (inData, outData) = mnistReader.LoadData(
            Path.Combine("mnist", "t10k-images.idx3-ubyte"),
            Path.Combine("mnist", "t10k-labels.idx1-ubyte"));

        var filePath = Path.Combine("mnist", "new-trained-model.3n");
        if (File.Exists(filePath))
        {
            var bytes = File.ReadAllBytes(filePath);
            var nn = NeuralNetworkSerialiser.Deserialise(bytes);
            Console.WriteLine("New model loaded successfully!");
            Console.WriteLine($"Testing on first 20 samples:\n");
            TryAllData(inData, outData, nn, 20);
        }
        else
        {
            Console.WriteLine("ERROR: New model not found!");
        }
    }

    private static void VerifyNNOnData()
    {
        var mnistReader = new MnistReader();
        var (inData, outData) = mnistReader.LoadData(
            Path.Combine("mnist", "t10k-images.idx3-ubyte"),
            Path.Combine("mnist", "t10k-labels.idx1-ubyte"));

        var filePath = Path.Combine("mnist", "pre-trained-mnist-model.3n");
        if (File.Exists(filePath))
        {
            var bytes = File.ReadAllBytes(filePath);
            var nn = NeuralNetworkSerialiser.Deserialise(bytes);
            TryAllData(inData, outData, nn, 100);
        }
    }

    [SupportedOSPlatform("windows")]
    private static void SplitMnistData()
    {
        var splitter = new MnistSplitter();
        splitter.SaveDataAsFiles(
            Path.Combine("mnist", "t10k-images.idx3-ubyte"),
            Path.Combine("mnist", "t10k-labels.idx1-ubyte"),
            "t10k-dataset",
            Path.Combine("t10k-dataset", "labels.csv"));
    }

    private static ThreeN.NeuralNetwork.NeuralNetwork Train(Matrix<float> inData, Matrix<float> outData, string? baseModelPath, string destinationModelPath, int samplesPerBath = 100, int? numberOfBatches = null)
    {
        ThreeN.NeuralNetwork.NeuralNetwork nn;
        if (!string.IsNullOrEmpty(baseModelPath) && File.Exists(baseModelPath))
        {
            var bytes = File.ReadAllBytes(baseModelPath);
            nn = NeuralNetworkSerialiser.Deserialise(bytes);
        }
        else
        {
            // New fluent API with He initialization (good for ReLU)
            nn = new NeuralNetworkBuilder()
                .WithInputs(784)
                .WithHiddenLayer(80, Activation.Relu)
                .WithOutputLayer(10, Activation.Sigmoid)
                .WithInitialization(WeightInitialization.He)
                .Build();
        }

        var gradient = Gradient.CreateFor(nn);

        var n = numberOfBatches ?? inData.Rows / samplesPerBath; // todo: handle remainder
        var prevRunTime = 0.0;

        for (int i = 0; i < n; i++)
        {
            var batchInData = new Matrix<float>(samplesPerBath, inData.Columns, i * samplesPerBath * inData.Columns, inData.Stride, inData.Data);
            var batchOutData = new Matrix<float>(samplesPerBath, outData.Columns, i * samplesPerBath * outData.Columns, outData.Stride, outData.Data);

            var estimatedTimeLeft = TimeSpan.FromMilliseconds(prevRunTime * (n - i));

            prevRunTime = ProcessNN($"MNIST {i}/{numberOfBatches} [ETA in:{estimatedTimeLeft}]", batchInData, batchOutData, nn, gradient, 100, 1e-1f);
        }

        var model = NeuralNetworkSerialiser.Serialise(nn);
        File.WriteAllBytes(destinationModelPath, model);

        return nn;
    }

    private static double ProcessNN(string name, Matrix<float> inData, Matrix<float> outData, ThreeN.NeuralNetwork.NeuralNetwork nn, Gradient gradient, int epochs, float learningRate)
    {
        Console.WriteLine($"--------------------------------{name}--------------------------------");

        var cost = nn.ComputeCost(inData, outData);
        Console.WriteLine($"Pre-training cost: {cost}");

        var sw = Stopwatch.StartNew();

        for (int i = 0; i < epochs; i++)
        {
            NeuralNetworkExtensions.BackPropagation(nn, gradient, inData, outData, l2Lambda: 0f, lowPenetration: true);
            nn.ApplyGradient(gradient, learningRate);
        }

        sw.Stop();
        cost = nn.ComputeCost(inData, outData);
        Console.WriteLine($"Post-training cost: {cost}; Elapsed: {sw.Elapsed.TotalMilliseconds}ms");

        Console.WriteLine("----------------------------------------------------------------------------------");
        Console.WriteLine();

        return sw.Elapsed.TotalMilliseconds;
    }

    static void TryAllData(Matrix<float> inData, Matrix<float> outData, ThreeN.NeuralNetwork.NeuralNetwork nn, int numOfSamples)
    {
        for (int i = 0; i < numOfSamples; i++)
        {
            inData.CopyRow(nn.InputLayer, i);

            nn.Forward();

            Console.Write($"({i}) Expected: {OneHotToValue(outData, i)}; \t Result: ");
            Print(nn.OutputLayer);
        }
    }

    private static int OneHotToValue(Matrix<float> matrix, int row)
    {
        for (int i = 0; i < matrix.Columns; i++)
        {
            if (matrix[row, i] == 1)
                return i;
        }

        return -1;
    }

    private static void Print<T>(Matrix<T> matrix) where T : INumber<T>
    {
        for (int j = 0; j < matrix.Columns; j++)
        {
            Console.Write($"{j} = {matrix[0, j]:F4} \t");
        }

        Console.WriteLine();
    }
}