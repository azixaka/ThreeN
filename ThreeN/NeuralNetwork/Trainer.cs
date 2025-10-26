using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork.Optimizers;

namespace ThreeN.NeuralNetwork;

/// <summary>
/// High-level training coordinator with validation and early stopping support.
/// </summary>
public sealed class Trainer
{
    private readonly NeuralNetwork _network;
    private readonly Gradient _gradient;
    private readonly IOptimizer _optimizer;
    private readonly LearningRateSchedule _schedule;

    public float L2Lambda { get; set; } = 0f;
    public int EarlyStoppingPatience { get; set; } = 0; // 0 = disabled
    public TrainingHistory History { get; } = new();

    /// <summary>
    /// Creates a trainer with a custom optimizer.
    /// </summary>
    /// <param name="network">The neural network to train.</param>
    /// <param name="optimizer">The optimizer to use (e.g., Adam, Momentum, SGD).</param>
    /// <param name="schedule">Optional learning rate schedule (defaults to constant).</param>
    public Trainer(NeuralNetwork network, IOptimizer optimizer,
        LearningRateSchedule? schedule = null)
    {
        _network = network;
        _gradient = Gradient.CreateFor(network);
        _optimizer = optimizer;
        _schedule = schedule ?? new ConstantSchedule(optimizer.LearningRate);
    }

    /// <summary>
    /// Creates a trainer with simple SGD optimizer.
    /// </summary>
    /// <param name="network">The neural network to train.</param>
    /// <param name="learningRate">Learning rate for SGD (typical values: 0.001-0.1).</param>
    /// <param name="schedule">Optional learning rate schedule (defaults to constant).</param>
    /// <example>
    /// <code>
    /// // Simple trainer with fixed learning rate
    /// var trainer = new Trainer(network, learningRate: 0.001f);
    /// trainer.Train(trainInputs, trainOutputs, epochs: 100);
    /// </code>
    /// </example>
    public Trainer(NeuralNetwork network, float learningRate,
        LearningRateSchedule? schedule = null)
        : this(network, new SGDOptimizer(learningRate), schedule)
    {
    }

    public void Train(Matrix<float> trainInputs, Matrix<float> trainOutputs,
        Matrix<float>? valInputs = null, Matrix<float>? valOutputs = null,
        int epochs = 100)
    {
        int bestEpoch = 0;
        float bestValLoss = float.MaxValue;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Update learning rate
            _optimizer.LearningRate = _schedule.GetLearningRate(epoch);

            // Training step
            NeuralNetworkExtensions.BackPropagation(_network, _gradient,
                trainInputs, trainOutputs, L2Lambda);
            _optimizer.Update(_network, _gradient);

            // Record metrics
            float trainLoss = _network.ComputeCost(trainInputs, trainOutputs, L2Lambda);
            float trainAcc = _network.ComputeAccuracy(trainInputs, trainOutputs);

            History.TrainLoss.Add(trainLoss);
            History.TrainAccuracy.Add(trainAcc);

            if (valInputs.HasValue && valOutputs.HasValue)
            {
                float valLoss = _network.ComputeCost(valInputs.Value, valOutputs.Value, L2Lambda);
                float valAcc = _network.ComputeAccuracy(valInputs.Value, valOutputs.Value);

                History.ValLoss.Add(valLoss);
                History.ValAccuracy.Add(valAcc);

                // Early stopping
                if (EarlyStoppingPatience > 0 && valLoss < bestValLoss)
                {
                    bestValLoss = valLoss;
                    bestEpoch = epoch;
                }
                else if (EarlyStoppingPatience > 0 &&
                    epoch - bestEpoch >= EarlyStoppingPatience)
                {
                    Console.WriteLine($"Early stopping at epoch {epoch} (best: {bestEpoch})");
                    break;
                }
            }
        }
    }
}
