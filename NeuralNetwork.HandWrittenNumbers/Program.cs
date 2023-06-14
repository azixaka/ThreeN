using System.Diagnostics;
using System.Numerics;
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;

namespace NeuralNetwork.HandWrittenNumbers;

sealed class Program
{
    static void Main()
    {
        //Train(inData, outData, "mnist\\pre-trained-mnist-model.3n", $"mnist\\{DateTime.UtcNow.Ticks}.3n", 100, 5);
        //SplitMnistData();
        VerifyNNOnData();
    }

    private static void VerifyNNOnData()
    {
        var mnistReader = new MnistReader();
        var (inData, outData) = mnistReader.LoadData("mnist\\t10k-images.idx3-ubyte", "mnist\\t10k-labels.idx1-ubyte");

        var filePath = "mnist\\pre-trained-mnist-model.3n";
        if (File.Exists(filePath))
        {
            var bytes = File.ReadAllBytes(filePath);
            var nn = NeuralNetworkSerialiser.Deserialise(bytes);
            TryAllData(inData, outData, nn, 100);
        }
    }

    private static void SplitMnistData()
    {
        var splitter = new MnistSplitter();
        splitter.SaveDataAsFiles("mnist\\t10k-images.idx3-ubyte", "mnist\\t10k-labels.idx1-ubyte", "t10k-dataset", "t10k-dataset\\labels.csv");
    }

    private static ThreeN.NeuralNetwork.NeuralNetwork Train(Matrix<float> inData, Matrix<float> outData, string baseModelPath, string destinationModelPath, int samplesPerBath = 100, int? numberOfBatches = null)
    {
        var configuration = new[] { 784, 80, 10 };
        var activations = new[] { ActivationFunctionType.Relu, ActivationFunctionType.Sigmoid };

        ThreeN.NeuralNetwork.NeuralNetwork nn;
        if (!string.IsNullOrEmpty(baseModelPath) && File.Exists(baseModelPath))
        {
            var bytes = File.ReadAllBytes(baseModelPath);
            nn = NeuralNetworkSerialiser.Deserialise(bytes);
        }
        else
        {
            nn = ThreeN.NeuralNetwork.NeuralNetwork.Create(activations, configuration);
            NeuralNetworkExtensions.HeInitialise(nn);
        }

        var nng = ThreeN.NeuralNetwork.NeuralNetwork.Create(activations, configuration);

        var n = numberOfBatches ?? inData.Rows / samplesPerBath; // todo: handle remainder
        var prevRunTime = 0.0;

        for (int i = 0; i < n; i++)
        {
            var batchInData = new Matrix<float>(samplesPerBath, inData.Columns, i * samplesPerBath * inData.Columns, inData.Stride, inData.Data);
            var batchOutData = new Matrix<float>(samplesPerBath, outData.Columns, i * samplesPerBath * outData.Columns, outData.Stride, outData.Data);

            var estimatedTimeLeft = TimeSpan.FromMilliseconds(prevRunTime * (n - i));

            prevRunTime = ProcessNN($"MNIST {i}/{numberOfBatches} [ETA in:{estimatedTimeLeft}]", batchInData, batchOutData, nn, nng, 100, 1e-1f);
        }

        var model = NeuralNetworkSerialiser.Serialise(nn);
        File.WriteAllBytes(destinationModelPath, model);

        return nn;
    }

    private static double ProcessNN(string name, Matrix<float> inData, Matrix<float> outData, ThreeN.NeuralNetwork.NeuralNetwork nn, ThreeN.NeuralNetwork.NeuralNetwork gradient, int epochs, float learningRate)
    {
        Console.WriteLine($"--------------------------------{name}--------------------------------");

        var cost = NeuralNetworkExtensions.Cost(nn, inData, outData);
        Console.WriteLine($"Pre-training cost: {cost}");

        var sw = Stopwatch.StartNew();

        for (int i = 0; i < epochs; i++)
        {
            NeuralNetworkExtensions.BackPropagation(nn, gradient, inData, outData, true);
            NeuralNetworkExtensions.Train(nn, gradient, learningRate);
        }

        sw.Stop();
        cost = NeuralNetworkExtensions.Cost(nn, inData, outData);
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

            NeuralNetworkExtensions.Forward(nn);

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