using ThreeN.NeuralNetwork;
using Xunit;

namespace ThreeN.Tests;

public class GradientTests
{
    [Fact]
    public void CreateFor_WithValidNetwork_CreatesMatchingGradient()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(10)
            .WithHiddenLayer(5, ActivationFunctionType.Relu)
            .WithOutputLayer(2, ActivationFunctionType.Sigmoid)
            .Build();

        // Act
        var gradient = Gradient.CreateFor(network);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(network.Weights.Length, gradient.Weights.Length);
        Assert.Equal(network.Biases.Length, gradient.Biases.Length);

        // Check dimensions match
        for (int i = 0; i < network.Weights.Length; i++)
        {
            Assert.Equal(network.Weights[i].Rows, gradient.Weights[i].Rows);
            Assert.Equal(network.Weights[i].Columns, gradient.Weights[i].Columns);
            Assert.Equal(network.Biases[i].Rows, gradient.Biases[i].Rows);
            Assert.Equal(network.Biases[i].Columns, gradient.Biases[i].Columns);
        }
    }

    [Fact]
    public void Fill_WithValue_FillsAllGradients()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(3)
            .WithHiddenLayer(2, ActivationFunctionType.Relu)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();
        var gradient = Gradient.CreateFor(network);

        // Act
        gradient.Fill(0.5f);

        // Assert
        for (int i = 0; i < gradient.Weights.Length; i++)
        {
            for (int j = 0; j < gradient.Weights[i].Rows; j++)
            {
                for (int k = 0; k < gradient.Weights[i].Columns; k++)
                {
                    Assert.Equal(0.5f, gradient.Weights[i][j, k]);
                }
            }

            for (int j = 0; j < gradient.Biases[i].Rows; j++)
            {
                for (int k = 0; k < gradient.Biases[i].Columns; k++)
                {
                    Assert.Equal(0.5f, gradient.Biases[i][j, k]);
                }
            }
        }
    }

    [Fact]
    public void Fill_WithZero_ResetGradients()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(3)
            .WithHiddenLayer(2, ActivationFunctionType.Relu)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();
        var gradient = Gradient.CreateFor(network);
        gradient.Fill(1.0f); // Fill with non-zero first

        // Act
        gradient.Fill(0f);

        // Assert
        for (int i = 0; i < gradient.Weights.Length; i++)
        {
            for (int j = 0; j < gradient.Weights[i].Rows; j++)
            {
                for (int k = 0; k < gradient.Weights[i].Columns; k++)
                {
                    Assert.Equal(0f, gradient.Weights[i][j, k]);
                }
            }
        }
    }
}
