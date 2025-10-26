namespace ThreeN.NeuralNetwork;

/// <summary>
/// Learning rate scheduling strategies.
/// </summary>
public abstract class LearningRateSchedule
{
    /// <summary>Computes learning rate for given epoch.</summary>
    public abstract float GetLearningRate(int epoch);
}

/// <summary>Constant learning rate (no decay).</summary>
public sealed class ConstantSchedule : LearningRateSchedule
{
    private readonly float _rate;

    public ConstantSchedule(float learningRate) => _rate = learningRate;

    public override float GetLearningRate(int epoch) => _rate;
}

/// <summary>Step decay: lr = initial * factor^(epoch / stepSize).</summary>
public sealed class StepDecaySchedule : LearningRateSchedule
{
    private readonly float _initial;
    private readonly float _factor;
    private readonly int _stepSize;

    public StepDecaySchedule(float initialRate, float decayFactor = 0.5f, int stepSize = 10)
    {
        _initial = initialRate;
        _factor = decayFactor;
        _stepSize = stepSize;
    }

    public override float GetLearningRate(int epoch)
        => _initial * (float)Math.Pow(_factor, epoch / _stepSize);
}

/// <summary>Exponential decay: lr = initial * e^(-decay * epoch).</summary>
public sealed class ExponentialDecaySchedule : LearningRateSchedule
{
    private readonly float _initial;
    private readonly float _decay;

    public ExponentialDecaySchedule(float initialRate, float decayRate = 0.01f)
    {
        _initial = initialRate;
        _decay = decayRate;
    }

    public override float GetLearningRate(int epoch)
        => _initial * (float)Math.Exp(-_decay * epoch);
}
