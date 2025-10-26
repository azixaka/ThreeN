using System.Threading.Tasks;
using ThreeN.LinearAlgebra;

namespace ThreeN.NeuralNetwork;

public static class NeuralNetworkExtensions
{
    /// <summary>
    /// Computes gradients via backpropagation with optional parallelization for large batches.
    /// </summary>
    /// <param name="l2Lambda">L2 regularization strength (0 = no regularization).</param>
    /// <remarks>
    /// Parallelizes batch processing when n >= 100 samples.
    /// Uses thread-local gradients and networks to avoid contention.
    /// Expected speedup: 2-8x on multi-core CPUs for large batches.
    /// </remarks>
    public static void BackPropagation(NeuralNetwork nn, Gradient gradient, Matrix<float> inData, Matrix<float> outData, float l2Lambda = 0f, bool lowPenetration = true)
    {
        if (inData.Rows != outData.Rows)
            throw new ArgumentException("Number of samples in input and output data must be equal");

        if (outData.Columns != nn.OutputLayer.Columns)
            throw new ArgumentException("Number of output neurons must be equal to number of columns in output data");

        var n = inData.Rows; // number of samples

        // Threshold: only parallelize if benefit > overhead
        const int PARALLEL_THRESHOLD = 100;

        if (n < PARALLEL_THRESHOLD)
        {
            // Sequential for small batches (avoid threading overhead)
            BackPropagationSequential(nn, gradient, inData, outData, l2Lambda, lowPenetration);
        }
        else
        {
            // Parallel for large batches
            BackPropagationParallel(nn, gradient, inData, outData, l2Lambda, lowPenetration);
        }
    }

    /// <summary>
    /// Sequential backpropagation implementation.
    /// </summary>
    private static void BackPropagationSequential(NeuralNetwork nn, Gradient gradient, Matrix<float> inData, Matrix<float> outData, float l2Lambda, bool lowPenetration)
    {
        var n = inData.Rows;
        var propagationMultiplier = lowPenetration ? 1 : 2;

        gradient.Fill(0);

        for (int i = 0; i < n; i++)
        {
            inData.CopyRow(nn.InputLayer, i);
            nn.Forward();

            for (int j = 0; j < nn.Activations.Length; j++)
            {
                MatrixExtensions.Fill(ref gradient.Activations[j], 0);
            }

            for (int j = 0; j < outData.Columns; j++)
            {
                gradient.Activations[nn.Activations.Length - 1][0, j] = (lowPenetration ? 2 : 1) * (nn.OutputLayer[0, j] - outData[i, j]);
            }

            for (int l = nn.Weights.Length; l > 0; l--)
            {
                for (int j = 0; j < nn.Activations[l].Columns; j++)
                {
                    float a = nn.Activations[l][0, j];
                    float da = gradient.Activations[l][0, j];
                    var activationFunctionType = nn.ActivationFunctions[l - 1];
                    float qa = ActivationFunctions.Derivative(a, activationFunctionType);
                    float q = propagationMultiplier * da * qa;

                    gradient.Biases[l - 1][0, j] += q;

                    for (int k = 0; k < nn.Activations[l - 1].Columns; k++)
                    {
                        float pa = nn.Activations[l - 1][0, k];
                        float w = nn.Weights[l - 1][k, j];
                        gradient.Weights[l - 1][k, j] += q * pa;
                        gradient.Activations[l - 1][0, k] += q * w;
                    }
                }
            }
        }

        // Average gradients and add L2 regularization term
        for (int i = 0; i < gradient.Weights.Length; i++)
        {
            for (int j = 0; j < gradient.Weights[i].Rows; j++)
                for (int k = 0; k < gradient.Weights[i].Columns; k++)
                {
                    gradient.Weights[i][j, k] = gradient.Weights[i][j, k] / n + l2Lambda * nn.Weights[i][j, k];
                }

            for (int j = 0; j < gradient.Biases[i].Rows; j++)
                for (int k = 0; k < gradient.Biases[i].Columns; k++)
                {
                    gradient.Biases[i][j, k] /= n;
                }
        }
    }

    /// <summary>
    /// Parallel backpropagation implementation for large batches.
    /// </summary>
    private static void BackPropagationParallel(NeuralNetwork nn, Gradient gradient, Matrix<float> inData, Matrix<float> outData, float l2Lambda, bool lowPenetration)
    {
        var n = inData.Rows;
        var propagationMultiplier = lowPenetration ? 1 : 2;

        gradient.Fill(0);

        // Thread-local storage for gradients and networks
        var threadLocalData = new ThreadLocal<(NeuralNetwork localNN, Gradient localGrad)>(() =>
        {
            var layerSizes = new int[nn.Activations.Length];
            for (int i = 0; i < layerSizes.Length; i++)
                layerSizes[i] = nn.Activations[i].Columns;

            var localNN = NeuralNetwork.Create(nn.ActivationFunctions, layerSizes);
            var localGrad = Gradient.CreateFor(localNN);

            // Copy weights from main network to local network
            for (int i = 0; i < nn.Weights.Length; i++)
            {
                Array.Copy(nn.Weights[i].Data, localNN.Weights[i].Data, nn.Weights[i].Data.Length);
                Array.Copy(nn.Biases[i].Data, localNN.Biases[i].Data, nn.Biases[i].Data.Length);
            }

            localGrad.Fill(0);

            return (localNN, localGrad);
        }, trackAllValues: true);

        // Parallel gradient computation
        Parallel.For(0, n, i =>
        {
            var (localNN, localGrad) = threadLocalData.Value!;

            inData.CopyRow(localNN.InputLayer, i);
            localNN.Forward();

            for (int j = 0; j < localNN.Activations.Length; j++)
            {
                MatrixExtensions.Fill(ref localGrad.Activations[j], 0);
            }

            for (int j = 0; j < outData.Columns; j++)
            {
                localGrad.Activations[localNN.Activations.Length - 1][0, j] = (lowPenetration ? 2 : 1) * (localNN.OutputLayer[0, j] - outData[i, j]);
            }

            for (int l = localNN.Weights.Length; l > 0; l--)
            {
                for (int j = 0; j < localNN.Activations[l].Columns; j++)
                {
                    float a = localNN.Activations[l][0, j];
                    float da = localGrad.Activations[l][0, j];
                    var activationFunctionType = localNN.ActivationFunctions[l - 1];
                    float qa = ActivationFunctions.Derivative(a, activationFunctionType);
                    float q = propagationMultiplier * da * qa;

                    localGrad.Biases[l - 1][0, j] += q;

                    for (int k = 0; k < localNN.Activations[l - 1].Columns; k++)
                    {
                        float pa = localNN.Activations[l - 1][0, k];
                        float w = localNN.Weights[l - 1][k, j];
                        localGrad.Weights[l - 1][k, j] += q * pa;
                        localGrad.Activations[l - 1][0, k] += q * w;
                    }
                }
            }
        });

        // Merge thread-local gradients into main gradient
        foreach (var (_, localGrad) in threadLocalData.Values)
        {
            for (int layer = 0; layer < gradient.Weights.Length; layer++)
            {
                for (int i = 0; i < gradient.Weights[layer].Rows; i++)
                    for (int j = 0; j < gradient.Weights[layer].Columns; j++)
                        gradient.Weights[layer][i, j] += localGrad.Weights[layer][i, j];

                for (int i = 0; i < gradient.Biases[layer].Rows; i++)
                    for (int j = 0; j < gradient.Biases[layer].Columns; j++)
                        gradient.Biases[layer][i, j] += localGrad.Biases[layer][i, j];
            }
        }

        threadLocalData.Dispose();

        // Average gradients and add L2 regularization term
        for (int i = 0; i < gradient.Weights.Length; i++)
        {
            for (int j = 0; j < gradient.Weights[i].Rows; j++)
                for (int k = 0; k < gradient.Weights[i].Columns; k++)
                {
                    gradient.Weights[i][j, k] = gradient.Weights[i][j, k] / n + l2Lambda * nn.Weights[i][j, k];
                }

            for (int j = 0; j < gradient.Biases[i].Rows; j++)
                for (int k = 0; k < gradient.Biases[i].Columns; k++)
                {
                    gradient.Biases[i][j, k] /= n;
                }
        }
    }

    /// <summary>
    /// Computes gradients using finite difference method (for gradient checking).
    /// </summary>
    public static void FiniteDifference(NeuralNetwork nn, Gradient gradient, float eps, Matrix<float> inData, Matrix<float> outData)
    {
        float saved;
        float cost = nn.ComputeCost(inData, outData);

        for (int i = 0; i < nn.Weights.Length; i++)
        {
            for (int j = 0; j < nn.Weights[i].Rows; j++)
                for (int k = 0; k < nn.Weights[i].Columns; k++)
                {
                    saved = nn.Weights[i][j, k];

                    nn.Weights[i][j, k] += eps;
                    gradient.Weights[i][j, k] = (nn.ComputeCost(inData, outData) - cost) / eps;
                    nn.Weights[i][j, k] = saved;
                }

            for (int j = 0; j < nn.Biases[i].Rows; j++)
                for (int k = 0; k < nn.Biases[i].Columns; k++)
                {
                    saved = nn.Biases[i][j, k];

                    nn.Biases[i][j, k] += eps;
                    gradient.Biases[i][j, k] = (nn.ComputeCost(inData, outData) - cost) / eps;
                    nn.Biases[i][j, k] = saved;
                }
        }
    }
}
