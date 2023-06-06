using System.Diagnostics;
using System.Numerics;
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;

sealed class Program
{
    static void Main()
    {
        var rawData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var inData = new Matrix<float>(4, 2, 0, 3, rawData);
        var outData = new Matrix<float>(4, 1, 2, 3, rawData);

        Print(ref inData, "in");
        Print(ref outData, "out");

        var configuration = new[] { 2, 10, 10, 10, 1 };
        var nn = NeuralNetwork.Create(configuration);
        NeuralNetworkExtensions.Randomise(nn, 0, 1);

        TryAllData(inData, nn);

        var cost = NeuralNetworkExtensions.Cost(nn, inData, outData);
        Console.WriteLine($"Pre-training cost: {cost}");

        var nng = NeuralNetwork.Create(configuration);

        var sw = Stopwatch.StartNew();

        for (int i = 0; i < 100_000; i++)
        {
            //NeuralNetworkExtensions.FiniteDifference(nn, nng, 1e-3f, inData, outData);
            NeuralNetworkExtensions.BackPropagation(nn, nng, inData, outData);
            NeuralNetworkExtensions.Train(nn, nng, 1f);
        }

        sw.Stop();
        cost = NeuralNetworkExtensions.Cost(nn, inData, outData);
        Console.WriteLine($"Post-training cost: {cost}; Elapsed: {sw.Elapsed.TotalMilliseconds}ms");

        TryAllData(inData, nn);

        static void TryAllData(Matrix<float> inData, NeuralNetwork nn)
        {
            for (int i = 0; i < inData.Rows; i++)
            {
                inData.CopyRow(nn.InputLayer, i);

                NeuralNetworkExtensions.Forward(nn);

                for (int j = 0; j < inData.Columns; j++)
                {
                    Console.Write($"{inData.ElementAt(i, j)} ");
                }

                Console.WriteLine($"{nn.OutputLayer.ElementAt(0, 0)}");
            }
        }
    }

    private static void Print<T>(ref Matrix<T> matrix, string name) where T : INumber<T>
    {
        Console.WriteLine($"{name} = [");
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                Console.Write($"\t {matrix.ElementAt(i, j):F4} \t");
            }

            Console.WriteLine();
        }
        Console.WriteLine("]");
        Console.WriteLine();
    }
}