using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using ThreeN.NeuralNetwork.Optimizers;
using Xunit;

namespace ThreeN.Tests;

public class OptimizerTests
{
    [Fact]
    public void SGDOptimizer_UpdatesWeightsCorrectly()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();

        network.Fill(0.5f);
        var initialWeight = network.Weights[0][0, 0];

        var gradient = Gradient.CreateFor(network);
        gradient.Fill(0.1f); // Positive gradient

        var optimizer = new SGDOptimizer(learningRate: 0.1f);

        // Act
        optimizer.Update(network, gradient);

        // Assert - weight should decrease (gradient descent)
        Assert.True(network.Weights[0][0, 0] < initialWeight);
        Assert.Equal(initialWeight - 0.1f * 0.1f, network.Weights[0][0, 0], precision: 6);
    }

    [Fact]
    public void SGDOptimizer_ConvergesOnXOR()
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
            .WithHiddenLayer(4, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var gradient = Gradient.CreateFor(network);
        var optimizer = new SGDOptimizer(learningRate: 1f);

        var initialCost = network.ComputeCost(inputs, outputs);

        // Act - train for 1000 epochs
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
            optimizer.Update(network, gradient);
        }

        var finalCost = network.ComputeCost(inputs, outputs);

        // Assert - should converge
        Assert.True(finalCost < initialCost);
        Assert.True(finalCost < 0.1f, $"Should converge to low cost, got {finalCost}");
    }

    [Fact]
    public void SGDOptimizer_InvalidLearningRate_ThrowsException()
    {
        // Assert
        Assert.Throws<ArgumentException>(() => new SGDOptimizer(learningRate: 0f));
        Assert.Throws<ArgumentException>(() => new SGDOptimizer(learningRate: -0.1f));
    }

    [Fact]
    public void SGDOptimizer_CanChangeLearningRate()
    {
        // Arrange
        var optimizer = new SGDOptimizer(learningRate: 0.01f);

        // Act
        optimizer.LearningRate = 0.1f;

        // Assert
        Assert.Equal(0.1f, optimizer.LearningRate);
    }

    [Fact]
    public void SGDOptimizer_Reset_DoesNothing()
    {
        // Arrange
        var optimizer = new SGDOptimizer(learningRate: 0.01f);

        // Act & Assert - should not throw
        optimizer.Reset();
    }

    [Fact]
    public void AdamOptimizer_UpdatesWeightsCorrectly()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();

        network.Fill(0.5f);
        var initialWeight = network.Weights[0][0, 0];

        var gradient = Gradient.CreateFor(network);
        gradient.Fill(0.1f); // Positive gradient

        var optimizer = new AdamOptimizer(learningRate: 0.001f);

        // Act
        optimizer.Update(network, gradient);

        // Assert - weight should decrease (gradient descent)
        Assert.True(network.Weights[0][0, 0] < initialWeight);
    }

    [Fact]
    public void AdamOptimizer_ConvergesOnXOR()
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
            .WithHiddenLayer(4, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var gradient = Gradient.CreateFor(network);
        var optimizer = new AdamOptimizer(learningRate: 0.01f);

        var initialCost = network.ComputeCost(inputs, outputs);

        // Act - train for 1000 epochs
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
            optimizer.Update(network, gradient);
        }

        var finalCost = network.ComputeCost(inputs, outputs);

        // Assert - should converge
        Assert.True(finalCost < initialCost);
        Assert.True(finalCost < 0.1f, $"Adam should converge to low cost, got {finalCost}");
    }

    [Fact]
    public void AdamOptimizer_InvalidLearningRate_ThrowsException()
    {
        // Assert
        Assert.Throws<ArgumentException>(() => new AdamOptimizer(learningRate: 0f));
        Assert.Throws<ArgumentException>(() => new AdamOptimizer(learningRate: -0.1f));
    }

    [Fact]
    public void AdamOptimizer_Reset_ClearsState()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();

        var gradient = Gradient.CreateFor(network);
        gradient.Fill(0.1f);

        var optimizer = new AdamOptimizer(learningRate: 0.01f);

        // Act - update once to initialize state
        optimizer.Update(network, gradient);

        // Now reset
        optimizer.Reset();

        // After reset, next update should behave like first update
        var weight1 = network.Weights[0][0, 0];
        optimizer.Update(network, gradient);
        var weight2 = network.Weights[0][0, 0];

        // Assert - weights should have changed
        Assert.NotEqual(weight1, weight2);
    }

    [Fact]
    public void MomentumOptimizer_UpdatesWeightsCorrectly()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();

        network.Fill(0.5f);
        var initialWeight = network.Weights[0][0, 0];

        var gradient = Gradient.CreateFor(network);
        gradient.Fill(0.1f); // Positive gradient

        var optimizer = new MomentumOptimizer(learningRate: 0.1f, momentum: 0.9f);

        // Act
        optimizer.Update(network, gradient);

        // Assert - weight should decrease (gradient descent)
        Assert.True(network.Weights[0][0, 0] < initialWeight);
    }

    [Fact]
    public void MomentumOptimizer_ConvergesOnXOR()
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
            .WithHiddenLayer(4, ActivationFunctionType.Sigmoid)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var gradient = Gradient.CreateFor(network);
        var optimizer = new MomentumOptimizer(learningRate: 1f, momentum: 0.9f);

        var initialCost = network.ComputeCost(inputs, outputs);

        // Act - train for 1000 epochs
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
            optimizer.Update(network, gradient);
        }

        var finalCost = network.ComputeCost(inputs, outputs);

        // Assert - should converge
        Assert.True(finalCost < initialCost);
        Assert.True(finalCost < 0.1f, $"Momentum should converge to low cost, got {finalCost}");
    }

    [Fact]
    public void MomentumOptimizer_InvalidParameters_ThrowsException()
    {
        // Assert
        Assert.Throws<ArgumentException>(() => new MomentumOptimizer(learningRate: 0f));
        Assert.Throws<ArgumentException>(() => new MomentumOptimizer(learningRate: -0.1f));
        Assert.Throws<ArgumentException>(() => new MomentumOptimizer(momentum: -0.1f));
        Assert.Throws<ArgumentException>(() => new MomentumOptimizer(momentum: 1f));
    }

    [Fact]
    public void MomentumOptimizer_Reset_ClearsVelocity()
    {
        // Arrange
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
            .Build();

        var gradient = Gradient.CreateFor(network);
        gradient.Fill(0.1f);

        var optimizer = new MomentumOptimizer(learningRate: 0.1f, momentum: 0.9f);

        // Act - update once to build velocity
        optimizer.Update(network, gradient);

        // Now reset
        optimizer.Reset();

        // After reset, next update should behave like first update (no accumulated velocity)
        var weight1 = network.Weights[0][0, 0];
        optimizer.Update(network, gradient);
        var weight2 = network.Weights[0][0, 0];

        // Assert - weights should have changed
        Assert.NotEqual(weight1, weight2);
    }
}
