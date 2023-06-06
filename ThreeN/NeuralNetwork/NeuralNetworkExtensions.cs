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
                    nn.Weights[i].ElementAt(j, k) -= rate * gradient.Weights[i].ElementAt(j, k);
                }

            for (int j = 0; j < nn.Biases[i].Rows; j++)
                for (int k = 0; k < nn.Biases[i].Columns; k++)
                {
                    nn.Biases[i].ElementAt(j, k) -= rate * gradient.Biases[i].ElementAt(j, k);
                }
        }
    }

    public static void BackPropagation(NeuralNetwork nn, NeuralNetwork gradient, Matrix<float> inData, Matrix<float> outData)
    {
        Fill(gradient, 0);

        for (int i = 0; i < inData.Rows; i++)
        {
            inData.CopyRow(nn.InputLayer, i);
            Forward(nn);

            for (int j = 0; j < nn.Activations.Length; j++)
            {
                MatrixExtensions.Fill(ref gradient.Activations[j], 0);
            }

            for (int j = 0; j < outData.Columns; j++)
            {
                var output = nn.OutputLayer.ElementAt(0, j);
                gradient.OutputLayer.ElementAt(0, j) = 2 * (output - outData.ElementAt(i, j));
            }

            for (int l = nn.Weights.Length; l > 0; l--)
            {
                for (int j = 0; j < nn.Activations[l].Columns; j++)
                {
                    float a = nn.Activations[l].ElementAt(0, j);
                    float da = gradient.Activations[l].ElementAt(0, j);
                    float q = 2 * da * a * (1 - a);

                    gradient.Biases[l - 1].ElementAt(0, j) += q;

                    for (int k = 0; k < nn.Activations[l - 1].Columns; k++)
                    {
                        float pa = nn.Activations[l - 1].ElementAt(0, k);
                        float w = nn.Weights[l - 1].ElementAt(k, j);
                        gradient.Weights[l - 1].ElementAt(k, j) += q * pa;
                        gradient.Activations[l - 1].ElementAt(0, k) += q * w;
                    }
                }
            }
        }

        for (int i = 0; i < gradient.Weights.Length; i++)
        {
            for (int j = 0; j < gradient.Weights[i].Rows; j++)
                for (int k = 0; k < gradient.Weights[i].Columns; k++)
                {
                    gradient.Weights[i].ElementAt(j, k) /= inData.Rows;
                }

            for (int j = 0; j < gradient.Biases[i].Rows; j++)
                for (int k = 0; k < gradient.Biases[i].Columns; k++)
                {
                    gradient.Biases[i].ElementAt(j, k) /= inData.Rows;
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
                    saved = nn.Weights[i].ElementAt(j, k);

                    nn.Weights[i].ElementAt(j, k) += eps;
                    gradient.Weights[i].ElementAt(j, k) = (Cost(nn, inData, outData) - cost) / eps;
                    nn.Weights[i].ElementAt(j, k) = saved;
                }

            for (int j = 0; j < nn.Biases[i].Rows; j++)
                for (int k = 0; k < nn.Biases[i].Columns; k++)
                {
                    saved = nn.Biases[i].ElementAt(j, k);

                    nn.Biases[i].ElementAt(j, k) += eps;
                    gradient.Biases[i].ElementAt(j, k) = (Cost(nn, inData, outData) - cost) / eps;
                    nn.Biases[i].ElementAt(j, k) = saved;
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
                var output = nn.OutputLayer.ElementAt(0, j);
                var d = output - outData.ElementAt(i, j);
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
            MatrixExtensions.Activate(ref nn.Activations[i + 1], Activations.Activations.Sigmoid);
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
