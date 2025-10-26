using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using Xunit;

namespace ThreeN.Tests;

public class SerializationTests
{
    [Fact]
    public void RoundTrip_SimpleNetwork_PreservesStructure()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, Activation.Relu)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        var bytes = NeuralNetworkSerialiser.Serialise(network);
        var loaded = NeuralNetworkSerialiser.Deserialise(bytes);

        Assert.Equal(network.Weights.Length, loaded.Weights.Length);
        Assert.Equal(network.Biases.Length, loaded.Biases.Length);
        Assert.Equal(network.ActivationFunctions.Length, loaded.ActivationFunctions.Length);
    }

    [Fact]
    public void RoundTrip_PreservesWeights()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(2, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        // Set specific weights
        network.Weights[0][0, 0] = 1.5f;
        network.Weights[0][0, 1] = 2.5f;
        network.Weights[1][0, 0] = 3.5f;

        var bytes = NeuralNetworkSerialiser.Serialise(network);
        var loaded = NeuralNetworkSerialiser.Deserialise(bytes);

        Assert.Equal(1.5f, loaded.Weights[0][0, 0]);
        Assert.Equal(2.5f, loaded.Weights[0][0, 1]);
        Assert.Equal(3.5f, loaded.Weights[1][0, 0]);
    }

    [Fact]
    public void RoundTrip_PreservesBiases()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(2, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .Build();

        // Set specific biases (biases are 1xN matrices)
        network.Biases[0][0, 0] = 0.5f;
        network.Biases[0][0, 1] = 1.5f;
        network.Biases[1][0, 0] = 2.5f;

        var bytes = NeuralNetworkSerialiser.Serialise(network);
        var loaded = NeuralNetworkSerialiser.Deserialise(bytes);

        Assert.Equal(0.5f, loaded.Biases[0][0, 0]);
        Assert.Equal(1.5f, loaded.Biases[0][0, 1]);
        Assert.Equal(2.5f, loaded.Biases[1][0, 0]);
    }

    [Fact]
    public void RoundTrip_PreservesActivationFunctions()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(3)
            .WithHiddenLayer(4, Activation.Relu)
            .WithHiddenLayer(3, Activation.Tanh)
            .WithOutputLayer(2, Activation.Softmax)
            .Build();

        var bytes = NeuralNetworkSerialiser.Serialise(network);
        var loaded = NeuralNetworkSerialiser.Deserialise(bytes);

        Assert.Equal(Activation.Relu, loaded.ActivationFunctions[0]);
        Assert.Equal(Activation.Tanh, loaded.ActivationFunctions[1]);
        Assert.Equal(Activation.Softmax, loaded.ActivationFunctions[2]);
    }

    [Fact]
    public void RoundTrip_ProducesSameOutput()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(3, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var input = new Matrix<float>(1, 2, new float[] { 0.5f, 0.8f });

        // Get output from original network
        input.CopyRow(network.InputLayer, 0);
        network.Forward();
        float originalOutput = network.OutputLayer[0, 0];

        // Serialize and deserialize
        var bytes = NeuralNetworkSerialiser.Serialise(network);
        var loaded = NeuralNetworkSerialiser.Deserialise(bytes);

        // Get output from loaded network
        input.CopyRow(loaded.InputLayer, 0);
        loaded.Forward();
        float loadedOutput = loaded.OutputLayer[0, 0];

        Assert.Equal(originalOutput, loadedOutput, precision: 6);
    }

    [Fact]
    public void RoundTrip_MultipleForwardPasses_ProduceSameOutputs()
    {
        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(4, Activation.Relu)
            .WithOutputLayer(2, Activation.Softmax)
            .Build();

        var bytes = NeuralNetworkSerialiser.Serialise(network);
        var loaded = NeuralNetworkSerialiser.Deserialise(bytes);

        var testInputs = new[]
        {
            new float[] { 0f, 0f },
            new float[] { 1f, 0f },
            new float[] { 0f, 1f },
            new float[] { 1f, 1f }
        };

        foreach (var testInput in testInputs)
        {
            var input = new Matrix<float>(1, 2, testInput);

            input.CopyRow(network.InputLayer, 0);
            network.Forward();
            float orig1 = network.OutputLayer[0, 0];
            float orig2 = network.OutputLayer[0, 1];

            input.CopyRow(loaded.InputLayer, 0);
            loaded.Forward();
            float load1 = loaded.OutputLayer[0, 0];
            float load2 = loaded.OutputLayer[0, 1];

            Assert.Equal(orig1, load1, precision: 6);
            Assert.Equal(orig2, load2, precision: 6);
        }
    }
}
