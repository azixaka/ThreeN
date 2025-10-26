using System.Diagnostics;
using System.Numerics;
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using ThreeN.NeuralNetwork.Optimizers;

sealed class Program
{
    static void Main()
    {
        XorNN();
        LinearNN();

        Console.WriteLine("\n=== NEW ADVANCED FEATURES ===\n");
        XorWithAdamOptimizer();
        XorWithTrainerAndEarlyStopping();
    }

    // Sigmoid, Tanh -> Xavier
    // Relu -> He

    private static void XorNN()
    {
        var rawData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var inData = new Matrix<float>(rawData.Length / 3, 2, 0, 3, rawData);
        var outData = new Matrix<float>(rawData.Length / 3, 1, 2, 3, rawData);

        // New fluent API
        var nn = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(2, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .WithInitialization(WeightInitialization.Random)
            .Build();

        var gradient = Gradient.CreateFor(nn);

        ProcessNN("XOR", inData, outData, nn, gradient, 1_000_000, 1f);
    }

    private static void LinearNN()
    {
        var rawData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 2,
            2, 2, 4,
            3, 2, 5,
            4, 3, 7,
            5, 5, 10
        };

        var inData = new Matrix<float>(rawData.Length / 3, 2, 0, 3, rawData);
        var outData = new Matrix<float>(rawData.Length / 3, 1, 2, 3, rawData);

        // New fluent API with He initialization (good for ReLU)
        var nn = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(5, ActivationFunctionType.Relu)
            .WithOutputLayer(1, ActivationFunctionType.PassThrough)
            .WithInitialization(WeightInitialization.He)
            .Build();

        var gradient = Gradient.CreateFor(nn);

        ProcessNN("Linear", inData, outData, nn, gradient, 1_000_000, 1e-3f);
    }

    private static void ProcessNN(string name, Matrix<float> inData, Matrix<float> outData, NeuralNetwork nn, Gradient gradient, int epochs, float learningRate)
    {
        Console.WriteLine($"--------------------------------{name}--------------------------------");

        Print(ref inData, "in");
        Print(ref outData, "out");

        TryAllData(inData, nn);

        var cost = nn.ComputeCost(inData, outData);
        Console.WriteLine($"Pre-training cost: {cost}");

        var sw = Stopwatch.StartNew();

        for (int i = 0; i < epochs; i++)
        {
            //NeuralNetworkExtensions.FiniteDifference(nn, gradient, 1e-3f, inData, outData);
            NeuralNetworkExtensions.BackPropagation(nn, gradient, inData, outData, l2Lambda: 0f, lowPenetration: false);
            nn.ApplyGradient(gradient, learningRate);
        }

        sw.Stop();
        cost = nn.ComputeCost(inData, outData);
        Console.WriteLine($"Post-training cost: {cost}; Elapsed: {sw.Elapsed.TotalMilliseconds}ms");

        TryAllData(inData, nn);

        Console.WriteLine("----------------------------------------------------------------------------------");
        Console.WriteLine();
    }

    static void TryAllData(Matrix<float> inData, NeuralNetwork nn)
    {
        for (int i = 0; i < inData.Rows; i++)
        {
            inData.CopyRow(nn.InputLayer, i);

            nn.Forward();

            for (int j = 0; j < inData.Columns; j++)
            {
                Console.Write($"{inData[i, j]} ");
            }

            Console.WriteLine($"{nn.OutputLayer[0, 0]}");
        }
    }

    private static void Print<T>(ref Matrix<T> matrix, string name) where T : INumber<T>
    {
        Console.WriteLine($"{name} = [");
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                Console.Write($"\t {matrix[i, j]:F4} \t");
            }

            Console.WriteLine();
        }
        Console.WriteLine("]");
        Console.WriteLine();
    }

    private static void XorWithAdamOptimizer()
    {
        Console.WriteLine("--------------------------------XOR with Adam Optimizer--------------------------------");

        var rawData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var inData = new Matrix<float>(rawData.Length / 3, 2, 0, 3, rawData);
        var outData = new Matrix<float>(rawData.Length / 3, 1, 2, 3, rawData);

        var nn = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(4, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var gradient = Gradient.CreateFor(nn);
        var optimizer = new AdamOptimizer(learningRate: 0.01f);

        Console.WriteLine($"Pre-training cost: {nn.ComputeCost(inData, outData)}");

        var sw = Stopwatch.StartNew();
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            NeuralNetworkExtensions.BackPropagation(nn, gradient, inData, outData);
            optimizer.Update(nn, gradient);
        }
        sw.Stop();

        Console.WriteLine($"Post-training cost: {nn.ComputeCost(inData, outData)}; Elapsed: {sw.Elapsed.TotalMilliseconds}ms");
        Console.WriteLine($"Accuracy: {nn.ComputeAccuracy(inData, outData):P0}");

        TryAllData(inData, nn);
        Console.WriteLine("----------------------------------------------------------------------------------\n");
    }

    private static void XorWithTrainerAndEarlyStopping()
    {
        Console.WriteLine("--------------------------------XOR with Trainer (Adam + L2 + Schedule + Early Stopping)--------------------------------");

        var rawData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var trainInputs = new Matrix<float>(rawData.Length / 3, 2, 0, 3, rawData);
        var trainOutputs = new Matrix<float>(rawData.Length / 3, 1, 2, 3, rawData);

        // Use same data for validation (in real scenarios, use different data)
        var valInputs = trainInputs;
        var valOutputs = trainOutputs;

        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(10, ActivationFunctionType.Sigmoid) // Overparameterized
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var optimizer = new AdamOptimizer(learningRate: 0.01f);
        var schedule = new StepDecaySchedule(initialRate: 0.01f, decayFactor: 0.9f, stepSize: 100);

        var trainer = new Trainer(network, optimizer, schedule)
        {
            L2Lambda = 0.01f,           // L2 regularization to prevent overfitting
            EarlyStoppingPatience = 20  // Stop if validation loss doesn't improve for 20 epochs
        };

        Console.WriteLine($"Pre-training cost: {network.ComputeCost(trainInputs, trainOutputs)}");

        var sw = Stopwatch.StartNew();
        trainer.Train(trainInputs, trainOutputs, valInputs, valOutputs, epochs: 1000);
        sw.Stop();

        Console.WriteLine($"Post-training cost: {network.ComputeCost(trainInputs, trainOutputs)}; Elapsed: {sw.Elapsed.TotalMilliseconds}ms");
        Console.WriteLine($"Accuracy: {network.ComputeAccuracy(trainInputs, trainOutputs):P0}");
        Console.WriteLine($"Epochs trained: {trainer.History.TrainLoss.Count}");
        Console.WriteLine($"Final learning rate: {optimizer.LearningRate}");

        // Show training progression
        Console.WriteLine("\nTraining progression (first 10 and last 10 epochs):");
        for (int i = 0; i < Math.Min(10, trainer.History.TrainLoss.Count); i++)
        {
            Console.WriteLine($"  Epoch {i}: Loss={trainer.History.TrainLoss[i]:F6}, Acc={trainer.History.TrainAccuracy[i]:P0}");
        }

        if (trainer.History.TrainLoss.Count > 20)
        {
            Console.WriteLine("  ...");
            int start = trainer.History.TrainLoss.Count - 10;
            for (int i = start; i < trainer.History.TrainLoss.Count; i++)
            {
                Console.WriteLine($"  Epoch {i}: Loss={trainer.History.TrainLoss[i]:F6}, Acc={trainer.History.TrainAccuracy[i]:P0}");
            }
        }

        TryAllData(trainInputs, network);
        Console.WriteLine("----------------------------------------------------------------------------------\n");
    }
}
