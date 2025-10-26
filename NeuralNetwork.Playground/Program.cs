using System.Diagnostics;
using System.Numerics;
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using ThreeN.NeuralNetwork.Optimizers;

sealed class Program
{
    static void Main()
    {
        Console.WriteLine("=== SIMPLE API EXAMPLES ===\n");
        XorSimple();
        LinearRegressionSimple();

        Console.WriteLine("\n=== ADVANCED API EXAMPLES ===\n");
        XorWithAdamOptimizer();
        XorWithTrainerAndEarlyStopping();
    }

    // ========== SIMPLE API EXAMPLES (NEW!) ==========

    private static void XorSimple()
    {
        Console.WriteLine("--------------------------------XOR (Simple API)--------------------------------");

        var xorData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        // Use static factory methods for clarity
        var inputs = Matrix<float>.FromArrayStrided(4, 2, xorData, startIndex: 0, stride: 3);
        var outputs = Matrix<float>.FromArrayStrided(4, 1, xorData, startIndex: 2, stride: 3);

        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(2, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        Console.WriteLine($"Pre-training cost: {network.ComputeCost(inputs, outputs):F6}");

        var sw = Stopwatch.StartNew();

        // NEW SIMPLE API: One line training!
        network.Train(inputs, outputs, epochs: 100_000, learningRate: 1f);

        sw.Stop();
        Console.WriteLine($"Post-training cost: {network.ComputeCost(inputs, outputs):F6}; Time: {sw.Elapsed.TotalMilliseconds:F0}ms");

        TestNetwork(inputs, network);
        Console.WriteLine("------------------------------------------------------------------------------------\n");
    }

    private static void LinearRegressionSimple()
    {
        Console.WriteLine("--------------------------------Linear Regression (Simple API)--------------------------------");

        var trainingData = new float[]
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

        var inputs = Matrix<float>.FromArrayStrided(8, 2, trainingData, startIndex: 0, stride: 3);
        var outputs = Matrix<float>.FromArrayStrided(8, 1, trainingData, startIndex: 2, stride: 3);

        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(5, Activation.Relu)
            .WithOutputLayer(1, Activation.PassThrough)  // Linear output for regression
            .WithInitialization(WeightInitialization.He)
            .Build();

        Console.WriteLine($"Pre-training cost: {network.ComputeCost(inputs, outputs):F6}");

        var sw = Stopwatch.StartNew();

        // NEW SIMPLE API with progress callback
        network.Train(inputs, outputs, epochs: 10_000, learningRate: 0.001f,
            onEpochComplete: (epoch, loss) =>
            {
                if (epoch % 2000 == 0)
                    Console.WriteLine($"  Epoch {epoch,5}: Loss = {loss:F6}");
            });

        sw.Stop();
        Console.WriteLine($"Post-training cost: {network.ComputeCost(inputs, outputs):F6}; Time: {sw.Elapsed.TotalMilliseconds:F0}ms");

        TestNetwork(inputs, network);
        Console.WriteLine("------------------------------------------------------------------------------------\n");
    }

    // ========== ADVANCED API EXAMPLES ==========

    private static void XorWithAdamOptimizer()
    {
        Console.WriteLine("--------------------------------XOR with Adam Optimizer--------------------------------");

        var xorData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var inputs = new Matrix<float>(4, 2, 0, 3, xorData);
        var outputs = new Matrix<float>(4, 1, 2, 3, xorData);

        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(4, Activation.Relu)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.He)
            .Build();

        Console.WriteLine($"Pre-training cost: {network.ComputeCost(inputs, outputs):F6}");

        var optimizer = new AdamOptimizer(learningRate: 0.01f);
        var sw = Stopwatch.StartNew();

        // NEW API with optimizer
        network.Train(inputs, outputs, epochs: 10_000, optimizer: optimizer,
            onEpochComplete: (epoch, loss) =>
            {
                if (epoch % 2000 == 0)
                    Console.WriteLine($"  Epoch {epoch,5}: Loss = {loss:F6}");
            });

        sw.Stop();
        Console.WriteLine($"Post-training cost: {network.ComputeCost(inputs, outputs):F6}; Time: {sw.Elapsed.TotalMilliseconds:F0}ms");

        TestNetwork(inputs, network);
        Console.WriteLine("------------------------------------------------------------------------------------\n");
    }

    private static void XorWithTrainerAndEarlyStopping()
    {
        Console.WriteLine("--------------------------------XOR with Trainer (Full Features)--------------------------------");

        var xorData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var inputs = new Matrix<float>(4, 2, 0, 3, xorData);
        var outputs = new Matrix<float>(4, 1, 2, 3, xorData);

        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(8, Activation.Relu)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.He)
            .Build();

        Console.WriteLine($"Pre-training cost: {network.ComputeCost(inputs, outputs):F6}");

        // Use Trainer with all advanced features
        var trainer = new Trainer(
            network,
            new AdamOptimizer(learningRate: 0.01f),
            new StepDecaySchedule(initialRate: 0.01f, decayFactor: 0.9f, stepSize: 1000)
        )
        {
            L2Lambda = 0.001f,  // L2 regularization
            EarlyStoppingPatience = 0  // Disabled for this small dataset
        };

        var sw = Stopwatch.StartNew();
        trainer.Train(inputs, outputs, epochs: 10_000);
        sw.Stop();

        Console.WriteLine($"Post-training cost: {network.ComputeCost(inputs, outputs):F6}; Time: {sw.Elapsed.TotalMilliseconds:F0}ms");
        Console.WriteLine($"Final training loss: {trainer.History.TrainLoss.Last():F6}");
        Console.WriteLine($"Final training accuracy: {trainer.History.TrainAccuracy.Last():P2}");

        TestNetwork(inputs, network);
        Console.WriteLine("------------------------------------------------------------------------------------\n");
    }

    // ========== HELPERS ==========

    static void TestNetwork(Matrix<float> inputs, NeuralNetwork network)
    {
        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < inputs.Rows; i++)
        {
            inputs.CopyRow(network.InputLayer, i);
            network.Forward();

            Console.Write("  Input: [");
            for (int j = 0; j < inputs.Columns; j++)
            {
                Console.Write($"{inputs[i, j]:F1}");
                if (j < inputs.Columns - 1) Console.Write(", ");
            }
            Console.WriteLine($"] → Output: {network.OutputLayer[0, 0]:F4}");
        }
    }
}
