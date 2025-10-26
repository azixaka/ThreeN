using ThreeN.NeuralNetwork;
using Xunit;

namespace ThreeN.Tests;

public class LearningRateScheduleTests
{
    [Fact]
    public void ConstantSchedule_ReturnsConstantRate()
    {
        // Arrange
        var schedule = new ConstantSchedule(learningRate: 0.01f);

        // Act & Assert
        Assert.Equal(0.01f, schedule.GetLearningRate(epoch: 0));
        Assert.Equal(0.01f, schedule.GetLearningRate(epoch: 10));
        Assert.Equal(0.01f, schedule.GetLearningRate(epoch: 100));
        Assert.Equal(0.01f, schedule.GetLearningRate(epoch: 1000));
    }

    [Fact]
    public void StepDecaySchedule_DecaysAtSteps()
    {
        // Arrange
        var schedule = new StepDecaySchedule(initialRate: 1.0f, decayFactor: 0.5f, stepSize: 10);

        // Act & Assert
        Assert.Equal(1.0f, schedule.GetLearningRate(epoch: 0), precision: 5);
        Assert.Equal(1.0f, schedule.GetLearningRate(epoch: 9), precision: 5);
        Assert.Equal(0.5f, schedule.GetLearningRate(epoch: 10), precision: 5);
        Assert.Equal(0.5f, schedule.GetLearningRate(epoch: 19), precision: 5);
        Assert.Equal(0.25f, schedule.GetLearningRate(epoch: 20), precision: 5);
        Assert.Equal(0.125f, schedule.GetLearningRate(epoch: 30), precision: 5);
    }

    [Fact]
    public void ExponentialDecaySchedule_DecaysExponentially()
    {
        // Arrange
        var schedule = new ExponentialDecaySchedule(initialRate: 1.0f, decayRate: 0.1f);

        // Act & Assert
        float lr0 = schedule.GetLearningRate(epoch: 0);
        float lr10 = schedule.GetLearningRate(epoch: 10);
        float lr20 = schedule.GetLearningRate(epoch: 20);

        Assert.Equal(1.0f, lr0, precision: 5);
        Assert.True(lr10 < lr0, $"Learning rate should decay: lr0={lr0}, lr10={lr10}");
        Assert.True(lr20 < lr10, $"Learning rate should continue decaying: lr10={lr10}, lr20={lr20}");

        // Verify exponential formula: lr = initial * e^(-decay * epoch)
        float expectedLr10 = 1.0f * (float)Math.Exp(-0.1f * 10);
        Assert.Equal(expectedLr10, lr10, precision: 5);
    }

    [Fact]
    public void ExponentialDecaySchedule_SmallDecayRate()
    {
        // Arrange
        var schedule = new ExponentialDecaySchedule(initialRate: 0.1f, decayRate: 0.01f);

        // Act & Assert - should decay slowly
        float lr0 = schedule.GetLearningRate(epoch: 0);
        float lr10 = schedule.GetLearningRate(epoch: 10);

        Assert.Equal(0.1f, lr0, precision: 5);
        Assert.True(lr10 < lr0, "Learning rate should decay");
        Assert.True(lr10 > 0.09f, "Decay should be small with small decay rate");
    }
}
