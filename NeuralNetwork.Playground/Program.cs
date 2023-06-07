using System.Diagnostics;
using System.Numerics;
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;

sealed class Program
{
    static void Main()
    {
        XorNN();
        LinearNN();
    }

    //Sigmoid, Tahn -> Xavier
    //Relu -> He

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

        var configuration = new[] { 2, 2, 1 };
        var activations = new[] { ActivationFunctionType.Sigmoid, ActivationFunctionType.Sigmoid };

        var nn = NeuralNetwork.Create(activations, configuration);
        //NeuralNetworkExtensions.Randomise(nn, 0, 1);
        NeuralNetworkExtensions.XavierInitialise(nn);
        var nng = NeuralNetwork.Create(activations, configuration);

        ProcessNN("XOR", inData, outData, nn, nng, 100_000, 1f);
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

        var configuration = new[] { 2, 2, 1 };
        var activations = new[] { ActivationFunctionType.Relu, ActivationFunctionType.PassThrough };

        var nn = NeuralNetwork.Create(activations, configuration);
        //NeuralNetworkExtensions.Randomise(nn, -5, 5);
        NeuralNetworkExtensions.HeInitialise(nn);
        var nng = NeuralNetwork.Create(activations, configuration);

        ProcessNN("Linear", inData, outData, nn, nng, 100_000, 1e-3f);
    }

    private static void ProcessNN(string name, Matrix<float> inData, Matrix<float> outData, NeuralNetwork nn, NeuralNetwork gradient, int epochs, float learningRate)
    {
        Console.WriteLine($"--------------------------------{name}--------------------------------");

        Print(ref inData, "in");
        Print(ref outData, "out");

        TryAllData(inData, nn);

        var cost = NeuralNetworkExtensions.Cost(nn, inData, outData);
        Console.WriteLine($"Pre-training cost: {cost}");

        var sw = Stopwatch.StartNew();

        for (int i = 0; i < epochs; i++)
        {
            //NeuralNetworkExtensions.FiniteDifference(nn, gradient, 1e-3f, inData, outData);
            NeuralNetworkExtensions.BackPropagation(nn, gradient, inData, outData);
            NeuralNetworkExtensions.Train(nn, gradient, learningRate);
        }

        sw.Stop();
        cost = NeuralNetworkExtensions.Cost(nn, inData, outData);
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

            NeuralNetworkExtensions.Forward(nn);

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
}