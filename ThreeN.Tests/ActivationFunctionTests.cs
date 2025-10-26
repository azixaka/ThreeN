using ThreeN.NeuralNetwork;
using Xunit;

namespace ThreeN.Tests;

public class ActivationFunctionTests
{
    [Fact]
    public void Sigmoid_ReturnsValuesBetween0And1()
    {
        Assert.InRange(ActivationFunctions.Sigmoid(-10f), 0f, 1f);
        Assert.InRange(ActivationFunctions.Sigmoid(0f), 0f, 1f);
        Assert.InRange(ActivationFunctions.Sigmoid(10f), 0f, 1f);

        Assert.Equal(0.5f, ActivationFunctions.Sigmoid(0f), precision: 5);
    }

    [Fact]
    public void SigmoidDerivative_IsCorrect()
    {
        float y = ActivationFunctions.Sigmoid(2f);
        float derivative = ActivationFunctions.SigmoidDerivative(y);

        // Derivative should equal y * (1 - y)
        Assert.Equal(y * (1f - y), derivative, precision: 6);
        Assert.True(derivative > 0);
    }

    [Fact]
    public void Relu_ReturnsCorrectValues()
    {
        Assert.Equal(5f, ActivationFunctions.Relu(5f));
        Assert.Equal(0f, ActivationFunctions.Relu(0f));
        Assert.Equal(-0.02f, ActivationFunctions.Relu(-2f), precision: 5); // Leaky: -2 * 0.01
    }

    [Fact]
    public void ReluDerivative_IsCorrect()
    {
        Assert.Equal(1f, ActivationFunctions.ReluDerivative(5f));
        Assert.Equal(1f, ActivationFunctions.ReluDerivative(0f));
        Assert.Equal(0.01f, ActivationFunctions.ReluDerivative(-1f));
    }

    [Fact]
    public void Tanh_ReturnsValuesBetweenMinus1And1()
    {
        Assert.InRange(ActivationFunctions.Tanh(-10f), -1f, 1f);
        Assert.InRange(ActivationFunctions.Tanh(0f), -1f, 1f);
        Assert.InRange(ActivationFunctions.Tanh(10f), -1f, 1f);

        Assert.Equal(0f, ActivationFunctions.Tanh(0f), precision: 5);
    }

    [Fact]
    public void TanhDerivative_IsCorrect()
    {
        float y = ActivationFunctions.Tanh(1f);
        float derivative = ActivationFunctions.TanhDerivative(y);

        // Derivative should equal 1 - y²
        Assert.Equal(1f - y * y, derivative, precision: 6);
        Assert.True(derivative > 0);
    }

    [Fact]
    public void Sin_ReturnsValuesBetweenMinus1And1()
    {
        Assert.InRange(ActivationFunctions.Sin(0f), -1f, 1f);
        Assert.InRange(ActivationFunctions.Sin(MathF.PI / 2), -1f, 1f);
        Assert.InRange(ActivationFunctions.Sin(MathF.PI), -1f, 1f);

        Assert.Equal(0f, ActivationFunctions.Sin(0f), precision: 5);
        Assert.Equal(1f, ActivationFunctions.Sin(MathF.PI / 2), precision: 5);
    }

    [Fact]
    public void SinDerivative_IsCorrect()
    {
        float y = ActivationFunctions.Sin(MathF.PI / 4);
        float derivative = ActivationFunctions.SinDerivative(y);

        // Derivative should equal sqrt(1 - y²)
        float expected = MathF.Sqrt(1f - y * y);
        Assert.Equal(expected, derivative, precision: 6);
    }

    [Fact]
    public void PassThrough_ReturnsInputUnchanged()
    {
        Assert.Equal(0f, ActivationFunctions.PassThrough(0f));
        Assert.Equal(5.5f, ActivationFunctions.PassThrough(5.5f));
        Assert.Equal(-3.2f, ActivationFunctions.PassThrough(-3.2f));
    }

    [Fact]
    public void PassThroughDerivative_AlwaysReturns1()
    {
        Assert.Equal(1f, ActivationFunctions.PassThroughDerivative(0f));
        Assert.Equal(1f, ActivationFunctions.PassThroughDerivative(100f));
        Assert.Equal(1f, ActivationFunctions.PassThroughDerivative(-50f));
    }

    [Fact]
    public void ActivateSoftmax_ProducesValidProbabilityDistribution()
    {
        var values = new float[] { 1f, 2f, 3f, 4f };

        ActivationFunctions.ActivateSoftmax(values);

        // All values should be between 0 and 1
        foreach (var v in values)
            Assert.InRange(v, 0f, 1f);

        // Sum should equal 1
        float sum = values.Sum();
        Assert.Equal(1f, sum, precision: 6);

        // Higher inputs should produce higher probabilities
        Assert.True(values[3] > values[2]);
        Assert.True(values[2] > values[1]);
        Assert.True(values[1] > values[0]);
    }

    [Fact]
    public void ActivateSoftmax_WithLargeValues_IsNumericallyStable()
    {
        var values = new float[] { 1000f, 1001f, 1002f };

        ActivationFunctions.ActivateSoftmax(values);

        // Should not produce NaN or infinity
        foreach (var v in values)
        {
            Assert.False(float.IsNaN(v));
            Assert.False(float.IsInfinity(v));
            Assert.InRange(v, 0f, 1f);
        }

        float sum = values.Sum();
        Assert.Equal(1f, sum, precision: 5);
    }

    [Fact]
    public void Activate_DispatchesCorrectly()
    {
        float x = 2f;

        Assert.Equal(ActivationFunctions.Sigmoid(x), ActivationFunctions.Activate(x, Activation.Sigmoid));
        Assert.Equal(ActivationFunctions.Relu(x), ActivationFunctions.Activate(x, Activation.Relu));
        Assert.Equal(ActivationFunctions.Tanh(x), ActivationFunctions.Activate(x, Activation.Tanh));
        Assert.Equal(ActivationFunctions.Sin(x), ActivationFunctions.Activate(x, Activation.Sin));
        Assert.Equal(ActivationFunctions.PassThrough(x), ActivationFunctions.Activate(x, Activation.PassThrough));
    }

    [Fact]
    public void Derivative_DispatchesCorrectly()
    {
        float y = 0.7f;

        Assert.Equal(ActivationFunctions.SigmoidDerivative(y), ActivationFunctions.Derivative(y, Activation.Sigmoid));
        Assert.Equal(ActivationFunctions.ReluDerivative(y), ActivationFunctions.Derivative(y, Activation.Relu));
        Assert.Equal(ActivationFunctions.TanhDerivative(y), ActivationFunctions.Derivative(y, Activation.Tanh));
        Assert.Equal(ActivationFunctions.SinDerivative(y), ActivationFunctions.Derivative(y, Activation.Sin));
        Assert.Equal(ActivationFunctions.PassThroughDerivative(y), ActivationFunctions.Derivative(y, Activation.PassThrough));
    }
}
