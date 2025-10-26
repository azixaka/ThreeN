namespace ThreeN.NeuralNetwork;

/// <summary>
/// Records training metrics over epochs.
/// </summary>
public sealed class TrainingHistory
{
    public List<float> TrainLoss { get; } = new();
    public List<float> TrainAccuracy { get; } = new();
    public List<float> ValLoss { get; } = new();
    public List<float> ValAccuracy { get; } = new();
}
