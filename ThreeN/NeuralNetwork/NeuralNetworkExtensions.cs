using ThreeN.LinearAlgebra;

namespace ThreeN.NeuralNetwork;

public static class NeuralNetworkExtensions
{
    public static void Train(NeuralNetwork nn, NeuralNetwork gradient, float rate)
    {
        for (int i = 0; i < nn.Weights.Length; i++)
        {
            for (int j = 0; j < nn.Weights[i].Rows; j++)
                for (int k = 0; k < nn.Weights[i].Columns; k++)
                {
                    nn.Weights[i][j, k] -= rate * gradient.Weights[i][j, k];
                }

            for (int j = 0; j < nn.Biases[i].Rows; j++)
                for (int k = 0; k < nn.Biases[i].Columns; k++)
                {
                    nn.Biases[i][j, k] -= rate * gradient.Biases[i][j, k];
                }
        }
    }

    public static void BackPropagation(NeuralNetwork nn, NeuralNetwork gradient, Matrix<float> inData, Matrix<float> outData)
    {
        if (inData.Rows != outData.Rows)
            throw new ArgumentException("Number of samples in input and output data must be equal");

        if (outData.Columns != nn.OutputLayer.Columns)
            throw new ArgumentException("Number of output neurons must be equal to number of columns in output data");

        var n = inData.Rows; // number of samples

        // i - current sample
        // l - current layer
        // j - current activation
        // k - previous activation

        Fill(gradient, 0);

        for (int i = 0; i < n; i++)
        {
            inData.CopyRow(nn.InputLayer, i);
            Forward(nn);

            for (int j = 0; j < nn.Activations.Length; j++)
            {
                MatrixExtensions.Fill(ref gradient.Activations[j], 0);
            }

            for (int j = 0; j < outData.Columns; j++)
            {
                gradient.OutputLayer[0, j] = 2 * (nn.OutputLayer[0, j] - outData[i, j]);
            }

            for (int l = nn.Weights.Length; l > 0; l--)
            {
                for (int j = 0; j < nn.Activations[l].Columns; j++)
                {
                    float a = nn.Activations[l][0, j];
                    float da = gradient.Activations[l][0, j];
                    var activationFunctionType = nn.ActivationFunctions[l - 1];
                    float qa = ActivationFunctions.Derivative(a, activationFunctionType);
                    float q = da * qa;

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

        for (int i = 0; i < gradient.Weights.Length; i++)
        {
            for (int j = 0; j < gradient.Weights[i].Rows; j++)
                for (int k = 0; k < gradient.Weights[i].Columns; k++)
                {
                    gradient.Weights[i][j, k] /= n;
                }

            for (int j = 0; j < gradient.Biases[i].Rows; j++)
                for (int k = 0; k < gradient.Biases[i].Columns; k++)
                {
                    gradient.Biases[i][j, k] /= n;
                }
        }
    }

    public static void FiniteDifference(NeuralNetwork nn, NeuralNetwork gradient, float eps, Matrix<float> inData, Matrix<float> outData)
    {
        float saved;
        float cost = Cost(nn, inData, outData);

        for (int i = 0; i < nn.Weights.Length; i++)
        {
            for (int j = 0; j < nn.Weights[i].Rows; j++)
                for (int k = 0; k < nn.Weights[i].Columns; k++)
                {
                    saved = nn.Weights[i][j, k];

                    nn.Weights[i][j, k] += eps;
                    gradient.Weights[i][j, k] = (Cost(nn, inData, outData) - cost) / eps;
                    nn.Weights[i][j, k] = saved;
                }

            for (int j = 0; j < nn.Biases[i].Rows; j++)
                for (int k = 0; k < nn.Biases[i].Columns; k++)
                {
                    saved = nn.Biases[i][j, k];

                    nn.Biases[i][j, k] += eps;
                    gradient.Biases[i][j, k] = (Cost(nn, inData, outData) - cost) / eps;
                    nn.Biases[i][j, k] = saved;
                }
        }
    }

    public static float Cost(NeuralNetwork nn, Matrix<float> inData, Matrix<float> outData)
    {
        float cost = 0;

        for (int i = 0; i < inData.Rows; i++)
        {
            inData.CopyRow(nn.InputLayer, i);
            Forward(nn);

            for (int j = 0; j < outData.Columns; j++)
            {
                var d = nn.OutputLayer[0, j] - outData[i, j];
                cost += d * d;
            }
        }

        return cost / inData.Rows;
    }

    public static void Forward(NeuralNetwork nn)
    {
        for (int i = 0; i < nn.Weights.Length; i++)
        {
            MatrixExtensions.DotProduct(nn.Activations[i + 1], nn.Activations[i], nn.Weights[i]);
            nn.Activations[i + 1].Add(ref nn.Biases[i]);
            MatrixExtensions.Activate(ref nn.Activations[i + 1], nn.ActivationFunctions[i]);
        }
    }

    public static void Fill(NeuralNetwork nn, float value)
    {
        for (int i = 0; i < nn.Weights.Length; i++)
        {
            MatrixExtensions.Fill(ref nn.Weights[i], value);
            MatrixExtensions.Fill(ref nn.Biases[i], value);
            MatrixExtensions.Fill(ref nn.Activations[i], value);
        }

        MatrixExtensions.Fill(ref nn.Activations[nn.Weights.Length], value);
    }

    public static void Randomise(NeuralNetwork nn, float low, float high)
    {
        for (int i = 0; i < nn.Weights.Length; i++)
        {
            MatrixExtensions.Randomise(ref nn.Weights[i], low, high);
            MatrixExtensions.Randomise(ref nn.Biases[i], low, high);
        }
    }
}
