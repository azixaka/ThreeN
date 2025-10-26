using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using Xunit;

namespace ThreeN.Tests;

public class NeuralNetworkTests
{
    [Fact]
    public void Forward_WithSimpleNetwork_ComputesOutput()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(2, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        network.InputLayer[0, 0] = 0.5f;
        network.InputLayer[0, 1] = 0.3f;

        // Act
        network.Forward();

        // Assert - output should be between 0 and 1 (sigmoid)
        Assert.InRange(network.OutputLayer[0, 0], 0f, 1f);
    }

    [Fact]
    public void ComputeCost_WithPerfectPrediction_ReturnsZero()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithOutputLayer(1, Activation.PassThrough)
            .Build();

        // Set weights to identity (output = input sum)
        network.Weights[0][0, 0] = 1f;
        network.Weights[0][1, 0] = 0f;
        network.Biases[0][0, 0] = 0f;

        var inputs = new Matrix<float>(1, 2);
        inputs[0, 0] = 1f;
        inputs[0, 1] = 0f;

        var outputs = new Matrix<float>(1, 1);
        outputs[0, 0] = 1f;

        // Act
        var cost = network.ComputeCost(inputs, outputs);

        // Assert
        Assert.Equal(0f, cost, precision: 5);
    }

    [Fact]
    public void ComputeCost_WithPoorPrediction_ReturnsHighCost()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithOutputLayer(1, Activation.PassThrough)
            .Build();

        network.Fill(0f); // All zeros

        var inputs = new Matrix<float>(1, 2);
        inputs[0, 0] = 1f;
        inputs[0, 1] = 1f;

        var outputs = new Matrix<float>(1, 1);
        outputs[0, 0] = 10f; // Expected 10, will get 0

        // Act
        var cost = network.ComputeCost(inputs, outputs);

        // Assert - cost should be high
        Assert.True(cost > 50f);
    }

    [Fact]
    public void ApplyGradient_UpdatesWeights()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(2, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        var gradient = Gradient.CreateFor(network);
        gradient.Fill(0.1f); // Small gradient

        var originalWeight = network.Weights[0][0, 0];

        // Act
        network.ApplyGradient(gradient, learningRate: 0.1f);

        // Assert - weight should change
        Assert.NotEqual(originalWeight, network.Weights[0][0, 0]);
    }

    [Fact]
    public void Fill_WithValue_FillsAllWeightsAndBiases()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(2, Activation.Relu)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        // Act
        network.Fill(0.5f);

        // Assert
        for (int i = 0; i < network.Weights.Length; i++)
        {
            for (int j = 0; j < network.Weights[i].Rows; j++)
            {
                for (int k = 0; k < network.Weights[i].Columns; k++)
                {
                    Assert.Equal(0.5f, network.Weights[i][j, k]);
                }
            }

            for (int j = 0; j < network.Biases[i].Rows; j++)
            {
                for (int k = 0; k < network.Biases[i].Columns; k++)
                {
                    Assert.Equal(0.5f, network.Biases[i][j, k]);
                }
            }
        }
    }

    [Fact]
    public void InitializeHe_InitializesWeightsNonZero()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(10)
            .WithHiddenLayer(5, Activation.Relu)
            .WithOutputLayer(2, Activation.Sigmoid)
            .Build();

        network.Fill(0f); // Reset to zero first

        // Act
        network.InitializeHe();

        // Assert - should have non-zero weights
        bool hasNonZeroWeights = false;
        for (int i = 0; i < network.Weights[0].Rows; i++)
        {
            for (int j = 0; j < network.Weights[0].Columns; j++)
            {
                if (network.Weights[0][i, j] != 0f)
                {
                    hasNonZeroWeights = true;
                    break;
                }
            }
        }
        Assert.True(hasNonZeroWeights);

        // Biases should be zero
        for (int i = 0; i < network.Biases[0].Columns; i++)
        {
            Assert.Equal(0f, network.Biases[0][0, i]);
        }
    }

    [Fact]
    public void InitializeXavier_InitializesWeightsNonZero()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(10)
            .WithHiddenLayer(5, Activation.Sigmoid)
            .WithOutputLayer(2, Activation.Sigmoid)
            .Build();

        network.Fill(0f); // Reset to zero first

        // Act
        network.InitializeXavier();

        // Assert - should have non-zero weights
        bool hasNonZeroWeights = false;
        for (int i = 0; i < network.Weights[0].Rows; i++)
        {
            for (int j = 0; j < network.Weights[0].Columns; j++)
            {
                if (network.Weights[0][i, j] != 0f)
                {
                    hasNonZeroWeights = true;
                    break;
                }
            }
        }
        Assert.True(hasNonZeroWeights);
    }

    [Fact]
    public void Randomize_InitializesWeightsInRange()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(10)
            .WithHiddenLayer(5, Activation.Relu)
            .WithOutputLayer(2, Activation.Sigmoid)
            .Build();

        // Act
        network.Randomize(-1f, 1f);

        // Assert - all weights should be in range [-1, 1]
        for (int i = 0; i < network.Weights.Length; i++)
        {
            for (int j = 0; j < network.Weights[i].Rows; j++)
            {
                for (int k = 0; k < network.Weights[i].Columns; k++)
                {
                    Assert.InRange(network.Weights[i][j, k], -1f, 1f);
                }
            }
        }
    }

    [Fact]
    public void XorTraining_ConvergesToSolution()
    {
        // Arrange - XOR problem
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
            .WithHiddenLayer(4, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var gradient = Gradient.CreateFor(network);

        var initialCost = network.ComputeCost(inputs, outputs);

        // Act - train for a few epochs
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
            network.ApplyGradient(gradient, learningRate: 1f);
        }

        var finalCost = network.ComputeCost(inputs, outputs);

        // Assert - cost should decrease
        Assert.True(finalCost < initialCost, $"Cost should decrease: initial={initialCost}, final={finalCost}");
    }
}
