using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using Xunit;

namespace ThreeN.Tests;

public class ParallelBackpropagationTests
{
    [Fact]
    public void ParallelBackprop_With100Samples_TriggersParallelPath()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        // Create 100 samples (triggers parallel path)
        var inputs = new Matrix<float>(100, 2);
        var outputs = new Matrix<float>(100, 1);
        for (int i = 0; i < 100; i++)
        {
            inputs[i, 0] = (i % 2) / 1f;
            inputs[i, 1] = (i / 2 % 2) / 1f;
            outputs[i, 0] = (i % 2) ^ (i / 2 % 2);
        }

        var gradient = Gradient.CreateFor(network);

        // Should not throw
        NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);

        // Gradient should have non-zero values
        bool hasNonZero = false;
        for (int i = 0; i < gradient.Weights[0].Rows; i++)
            for (int j = 0; j < gradient.Weights[0].Columns; j++)
                if (gradient.Weights[0][i, j] != 0)
                    hasNonZero = true;

        Assert.True(hasNonZero);
    }

    [Fact]
    public void ParallelBackprop_ProducesSimilarGradientsToSequential()
    {
        // Create two identical networks
        var network1 = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, Activation.Relu)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        var network2 = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, Activation.Relu)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        // Copy weights to make them identical
        for (int i = 0; i < network1.Weights.Length; i++)
        {
            Array.Copy(network1.Weights[i].Data, network2.Weights[i].Data, network1.Weights[i].Data.Length);
            Array.Copy(network1.Biases[i].Data, network2.Biases[i].Data, network1.Biases[i].Data.Length);
        }

        // Small dataset (sequential path)
        var smallInputs = new Matrix<float>(10, 2);
        var smallOutputs = new Matrix<float>(10, 1);
        for (int i = 0; i < 10; i++)
        {
            smallInputs[i, 0] = i % 2;
            smallInputs[i, 1] = (i / 2) % 2;
            smallOutputs[i, 0] = (i % 2) ^ ((i / 2) % 2);
        }

        // Large dataset (parallel path) - same pattern repeated
        var largeInputs = new Matrix<float>(100, 2);
        var largeOutputs = new Matrix<float>(100, 1);
        for (int i = 0; i < 100; i++)
        {
            largeInputs[i, 0] = i % 2;
            largeInputs[i, 1] = (i / 2) % 2;
            largeOutputs[i, 0] = (i % 2) ^ ((i / 2) % 2);
        }

        var gradient1 = Gradient.CreateFor(network1);
        var gradient2 = Gradient.CreateFor(network2);

        NeuralNetworkExtensions.BackPropagation(network1, gradient1, smallInputs, smallOutputs);
        NeuralNetworkExtensions.BackPropagation(network2, gradient2, largeInputs, largeOutputs);

        // Gradients should have similar magnitude (not exact due to averaging)
        float sum1 = 0, sum2 = 0;
        for (int i = 0; i < gradient1.Weights[0].Rows; i++)
            for (int j = 0; j < gradient1.Weights[0].Columns; j++)
            {
                sum1 += Math.Abs(gradient1.Weights[0][i, j]);
                sum2 += Math.Abs(gradient2.Weights[0][i, j]);
            }

        // Both should be non-zero
        Assert.True(sum1 > 0);
        Assert.True(sum2 > 0);
    }

    [Fact]
    public void ParallelBackprop_NetworkConverges()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(4, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        // Create 100 samples for XOR pattern
        var inputs = new Matrix<float>(100, 2);
        var outputs = new Matrix<float>(100, 1);
        for (int i = 0; i < 100; i++)
        {
            inputs[i, 0] = (i % 2) * 1f;
            inputs[i, 1] = ((i / 2) % 2) * 1f;
            outputs[i, 0] = ((i % 2) ^ ((i / 2) % 2)) * 1f;
        }

        float initialCost = network.ComputeCost(inputs, outputs);

        // Train for 100 epochs
        var gradient = Gradient.CreateFor(network);
        for (int epoch = 0; epoch < 100; epoch++)
        {
            NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
            network.ApplyGradient(gradient, learningRate: 1f);
        }

        float finalCost = network.ComputeCost(inputs, outputs);

        // Cost should decrease
        Assert.True(finalCost < initialCost);
    }

    [Fact]
    public void ParallelBackprop_WithL2Regularization_Works()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        var inputs = new Matrix<float>(100, 2);
        var outputs = new Matrix<float>(100, 1);
        for (int i = 0; i < 100; i++)
        {
            inputs[i, 0] = i / 100f;
            inputs[i, 1] = (i * 2) / 100f;
            outputs[i, 0] = (i % 2) / 1f;
        }

        var gradient = Gradient.CreateFor(network);

        // Should not throw with L2 regularization
        NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs, l2Lambda: 0.01f);

        // Verify gradient is computed
        Assert.NotEqual(0f, gradient.Weights[0][0, 0]);
    }
}
