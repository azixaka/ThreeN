using ThreeN.NeuralNetwork;
using Xunit;

namespace ThreeN.Tests;

public class NeuralNetworkBuilderTests
{
    [Fact]
    public void Builder_WithValidConfiguration_CreatesNetwork()
    {
        // Act
        var network = new NeuralNetworkBuilder()
            .WithInputs(10)
            .WithHiddenLayer(5, Activation.Relu)
            .WithOutputLayer(2, Activation.Sigmoid)
            .Build();

        // Assert
        Assert.NotNull(network);
        Assert.Equal(10, network.InputLayer.Columns);
        Assert.Equal(2, network.OutputLayer.Columns);
        Assert.Equal(2, network.Weights.Length); // 2 layers (hidden + output)
    }

    [Fact]
    public void Builder_WithMultipleHiddenLayers_CreatesNetwork()
    {
        // Act
        var network = new NeuralNetworkBuilder()
            .WithInputs(784)
            .WithHiddenLayer(128, Activation.Relu)
            .WithHiddenLayer(64, Activation.Relu)
            .WithOutputLayer(10, Activation.Softmax)
            .Build();

        // Assert
        Assert.NotNull(network);
        Assert.Equal(784, network.InputLayer.Columns);
        Assert.Equal(10, network.OutputLayer.Columns);
        Assert.Equal(3, network.Weights.Length); // 3 layers
    }

    [Fact]
    public void Builder_WithoutInputs_ThrowsInvalidOperationException()
    {
        // Arrange
        var builder = new NeuralNetworkBuilder()
            .WithOutputLayer(10, Activation.Softmax);

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => builder.Build());
        Assert.Contains("WithInputs", ex.Message);
    }

    [Fact]
    public void Builder_WithoutLayers_ThrowsInvalidOperationException()
    {
        // Arrange
        var builder = new NeuralNetworkBuilder()
            .WithInputs(10);

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => builder.Build());
        Assert.Contains("at least one layer", ex.Message);
    }

    [Fact]
    public void Builder_WithZeroInputs_ThrowsArgumentException()
    {
        // Arrange
        var builder = new NeuralNetworkBuilder();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => builder.WithInputs(0));
    }

    [Fact]
    public void Builder_WithNegativeInputs_ThrowsArgumentException()
    {
        // Arrange
        var builder = new NeuralNetworkBuilder();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => builder.WithInputs(-5));
    }

    [Fact]
    public void Builder_WithHeInitialization_InitializesWeights()
    {
        // Act
        var network = new NeuralNetworkBuilder()
            .WithInputs(10)
            .WithHiddenLayer(5, Activation.Relu)
            .WithOutputLayer(2, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.He)
            .Build();

        // Assert - weights should not all be zero
        bool hasNonZeroWeights = false;
        for (int i = 0; i < network.Weights[0].Rows; i++)
        {
            for (int j = 0; j < network.Weights[0].Columns; j++)
            {
                if (network.Weights[0][i, j] != 0)
                {
                    hasNonZeroWeights = true;
                    break;
                }
            }
        }
        Assert.True(hasNonZeroWeights);
    }

    [Fact]
    public void Builder_WithXavierInitialization_InitializesWeights()
    {
        // Act
        var network = new NeuralNetworkBuilder()
            .WithInputs(10)
            .WithHiddenLayer(5, Activation.Sigmoid)
            .WithOutputLayer(2, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        // Assert - weights should not all be zero
        bool hasNonZeroWeights = false;
        for (int i = 0; i < network.Weights[0].Rows; i++)
        {
            for (int j = 0; j < network.Weights[0].Columns; j++)
            {
                if (network.Weights[0][i, j] != 0)
                {
                    hasNonZeroWeights = true;
                    break;
                }
            }
        }
        Assert.True(hasNonZeroWeights);
    }

    [Fact]
    public void Builder_FluentAPI_ReturnsBuilderForChaining()
    {
        // Act - this test verifies the fluent API works
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, Activation.Relu)
            .WithHiddenLayer(3, Activation.Relu)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.He)
            .Build();

        // Assert
        Assert.NotNull(network);
    }
}
