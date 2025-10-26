using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using Xunit;

namespace ThreeN.Tests;

public class RegularizationTests
{
    [Fact]
    public void L2Regularization_IncreasesGradientMagnitude()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();

        network.Randomize(-1f, 1f);

        var inputs = new Matrix<float>(2, 2);
        inputs[0, 0] = 0f; inputs[0, 1] = 1f;
        inputs[1, 0] = 1f; inputs[1, 1] = 0f;

        var outputs = new Matrix<float>(2, 1);
        outputs[0, 0] = 1f;
        outputs[1, 0] = 0f;

        var gradientNoReg = Gradient.CreateFor(network);
        var gradientWithReg = Gradient.CreateFor(network);

        // Act - compute gradients without and with regularization
        NeuralNetworkExtensions.BackPropagation(network, gradientNoReg, inputs, outputs, l2Lambda: 0f);
        NeuralNetworkExtensions.BackPropagation(network, gradientWithReg, inputs, outputs, l2Lambda: 0.1f);

        // Assert - L2 regularization should increase gradient magnitude (pushing towards zero)
        float gradSumNoReg = 0f;
        float gradSumWithReg = 0f;

        for (int i = 0; i < network.Weights.Length; i++)
        {
            for (int j = 0; j < network.Weights[i].Rows; j++)
            for (int k = 0; k < network.Weights[i].Columns; k++)
            {
                gradSumNoReg += Math.Abs(gradientNoReg.Weights[i][j, k]);
                gradSumWithReg += Math.Abs(gradientWithReg.Weights[i][j, k]);
            }
        }

        Assert.True(gradSumWithReg > gradSumNoReg,
            $"L2 regularization should increase gradient magnitude: noReg={gradSumNoReg}, withReg={gradSumWithReg}");
    }

    [Fact]
    public void L2Regularization_IncreasesComputedCost()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();

        network.Randomize(-1f, 1f);

        var inputs = new Matrix<float>(2, 2);
        inputs[0, 0] = 0f; inputs[0, 1] = 1f;
        inputs[1, 0] = 1f; inputs[1, 1] = 0f;

        var outputs = new Matrix<float>(2, 1);
        outputs[0, 0] = 1f;
        outputs[1, 0] = 0f;

        // Act
        float costNoReg = network.ComputeCost(inputs, outputs, l2Lambda: 0f);
        float costWithReg = network.ComputeCost(inputs, outputs, l2Lambda: 0.1f);

        // Assert - L2 regularization adds penalty to cost
        Assert.True(costWithReg > costNoReg,
            $"L2 regularization should increase cost: noReg={costNoReg}, withReg={costWithReg}");
    }

    [Fact]
    public void L2Regularization_ReducesWeightMagnitudes()
    {
        // Arrange
        var xorData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var inputs = new Matrix<float>(4, 2, 0, 3, xorData);
        var outputs = new Matrix<float>(4, 1, 2, 3, xorData);

        // Train two networks: one without regularization, one with
        var networkNoReg = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(10, ActivationFunctionType.Sigmoid) // Overparameterized
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var networkWithReg = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(10, ActivationFunctionType.Sigmoid) // Overparameterized
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        // Copy weights to ensure same initial conditions
        for (int i = 0; i < networkNoReg.Weights.Length; i++)
        {
            Array.Copy(networkNoReg.Weights[i].Data, networkWithReg.Weights[i].Data, networkNoReg.Weights[i].Data.Length);
            Array.Copy(networkNoReg.Biases[i].Data, networkWithReg.Biases[i].Data, networkNoReg.Biases[i].Data.Length);
        }

        var gradNoReg = Gradient.CreateFor(networkNoReg);
        var gradWithReg = Gradient.CreateFor(networkWithReg);

        // Act - train both networks
        for (int epoch = 0; epoch < 100; epoch++)
        {
            NeuralNetworkExtensions.BackPropagation(networkNoReg, gradNoReg, inputs, outputs, l2Lambda: 0f);
            networkNoReg.ApplyGradient(gradNoReg, learningRate: 0.5f);

            NeuralNetworkExtensions.BackPropagation(networkWithReg, gradWithReg, inputs, outputs, l2Lambda: 0.1f);
            networkWithReg.ApplyGradient(gradWithReg, learningRate: 0.5f);
        }

        // Assert - regularized network should have smaller weights
        float weightSumNoReg = 0f;
        float weightSumWithReg = 0f;

        for (int i = 0; i < networkNoReg.Weights.Length; i++)
        {
            for (int j = 0; j < networkNoReg.Weights[i].Rows; j++)
            for (int k = 0; k < networkNoReg.Weights[i].Columns; k++)
            {
                weightSumNoReg += Math.Abs(networkNoReg.Weights[i][j, k]);
                weightSumWithReg += Math.Abs(networkWithReg.Weights[i][j, k]);
            }
        }

        Assert.True(weightSumWithReg < weightSumNoReg,
            $"L2 regularization should reduce weight magnitudes: noReg={weightSumNoReg}, withReg={weightSumWithReg}");
    }
}
