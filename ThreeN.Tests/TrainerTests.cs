using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;
using ThreeN.NeuralNetwork.Optimizers;
using Xunit;

namespace ThreeN.Tests;

public class TrainerTests
{
    [Fact]
    public void Trainer_RecordsTrainingHistory()
    {
        // Arrange
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
            .WithHiddenLayer(4, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var optimizer = new SGDOptimizer(learningRate: 1f);
        var trainer = new Trainer(network, optimizer);

        // Act
        trainer.Train(inputs, outputs, epochs: 10);

        // Assert
        Assert.Equal(10, trainer.History.TrainLoss.Count);
        Assert.Equal(10, trainer.History.TrainAccuracy.Count);
        Assert.Empty(trainer.History.ValLoss);
        Assert.Empty(trainer.History.ValAccuracy);
    }

    [Fact]
    public void Trainer_ConvergesOnXOR()
    {
        // Arrange
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
            .WithHiddenLayer(4, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var optimizer = new AdamOptimizer(learningRate: 0.01f);
        var trainer = new Trainer(network, optimizer);

        // Act
        trainer.Train(inputs, outputs, epochs: 1000);

        // Assert - training loss should decrease
        Assert.True(trainer.History.TrainLoss.Last() < trainer.History.TrainLoss.First(),
            $"Training loss should decrease: initial={trainer.History.TrainLoss.First()}, final={trainer.History.TrainLoss.Last()}");

        // Should achieve reasonable accuracy
        Assert.True(trainer.History.TrainAccuracy.Last() > 0.5f,
            $"Should achieve reasonable accuracy, got {trainer.History.TrainAccuracy.Last()}");
    }

    [Fact]
    public void Trainer_WithValidationSet_RecordsValidationMetrics()
    {
        // Arrange
        var xorData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var trainInputs = new Matrix<float>(4, 2, 0, 3, xorData);
        var trainOutputs = new Matrix<float>(4, 1, 2, 3, xorData);
        var valInputs = new Matrix<float>(4, 2, 0, 3, xorData); // Same data for simplicity
        var valOutputs = new Matrix<float>(4, 1, 2, 3, xorData);

        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(4, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var optimizer = new SGDOptimizer(learningRate: 1f);
        var trainer = new Trainer(network, optimizer);

        // Act
        trainer.Train(trainInputs, trainOutputs, valInputs, valOutputs, epochs: 10);

        // Assert
        Assert.Equal(10, trainer.History.TrainLoss.Count);
        Assert.Equal(10, trainer.History.TrainAccuracy.Count);
        Assert.Equal(10, trainer.History.ValLoss.Count);
        Assert.Equal(10, trainer.History.ValAccuracy.Count);
    }

    [Fact]
    public void Trainer_WithEarlyStopping_StopsEarly()
    {
        // Arrange - Create training data
        var xorData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var trainInputs = new Matrix<float>(4, 2, 0, 3, xorData);
        var trainOutputs = new Matrix<float>(4, 1, 2, 3, xorData);

        // Create validation data with intentional errors to simulate validation loss not improving
        var valData = new float[]
        {
            0, 0, 1,  // Wrong output (should be 0)
            0, 1, 0,  // Wrong output (should be 1)
            1, 0, 0,  // Wrong output (should be 1)
            1, 1, 1,  // Wrong output (should be 0)
        };
        var valInputs = new Matrix<float>(4, 2, 0, 3, valData);
        var valOutputs = new Matrix<float>(4, 1, 2, 3, valData);

        var network = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(10, Activation.Sigmoid) // Overparameterized
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var optimizer = new SGDOptimizer(learningRate: 0.5f);
        var trainer = new Trainer(network, optimizer)
        {
            EarlyStoppingPatience = 10
        };

        // Act
        trainer.Train(trainInputs, trainOutputs, valInputs, valOutputs, epochs: 1000);

        // Assert - should stop before 1000 epochs due to early stopping
        // The validation data has wrong labels, so validation loss will increase while training loss decreases
        Assert.True(trainer.History.TrainLoss.Count < 1000,
            $"Early stopping should prevent all 1000 epochs, ran {trainer.History.TrainLoss.Count}");
    }

    [Fact]
    public void Trainer_WithL2Regularization_ReducesOverfitting()
    {
        // Arrange
        var xorData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var inputs = new Matrix<float>(4, 2, 0, 3, xorData);
        var outputs = new Matrix<float>(4, 1, 2, 3, xorData);

        var networkNoReg = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(10, Activation.Sigmoid) // Overparameterized
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var networkWithReg = new NeuralNetworkBuilder()
            .WithInputs(2)
            .WithHiddenLayer(10, Activation.Sigmoid) // Overparameterized
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        // Copy weights
        for (int i = 0; i < networkNoReg.Weights.Length; i++)
        {
            Array.Copy(networkNoReg.Weights[i].Data, networkWithReg.Weights[i].Data, networkNoReg.Weights[i].Data.Length);
            Array.Copy(networkNoReg.Biases[i].Data, networkWithReg.Biases[i].Data, networkNoReg.Biases[i].Data.Length);
        }

        var trainerNoReg = new Trainer(networkNoReg, new SGDOptimizer(learningRate: 0.5f));
        var trainerWithReg = new Trainer(networkWithReg, new SGDOptimizer(learningRate: 0.5f))
        {
            L2Lambda = 0.1f
        };

        // Act
        trainerNoReg.Train(inputs, outputs, epochs: 100);
        trainerWithReg.Train(inputs, outputs, epochs: 100);

        // Assert - regularized network should have smaller weights
        float weightSumNoReg = 0f;
        float weightSumWithReg = 0f;

        for (int i = 0; i < networkNoReg.Weights.Length; i++)
        {
            for (int j = 0; j < networkNoReg.Weights[i].Rows; j++)
            for (int k = 0; k < networkNoReg.Weights[i].Columns; k++)
            {
                weightSumNoReg += Math.Abs(networkNoReg.Weights[i][j, k]);
                weightSumWithReg += Math.Abs(networkWithReg.Weights[i][j, k]);
            }
        }

        Assert.True(weightSumWithReg < weightSumNoReg,
            $"L2 regularization should reduce weight magnitudes: noReg={weightSumNoReg}, withReg={weightSumWithReg}");
    }

    [Fact]
    public void Trainer_WithLearningRateSchedule_AdjustsLearningRate()
    {
        // Arrange
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
            .WithHiddenLayer(4, Activation.Sigmoid)
            .WithOutputLayer(1, Activation.Sigmoid)
            .WithInitialization(WeightInitialization.Xavier)
            .Build();

        var optimizer = new SGDOptimizer(learningRate: 1f);
        var schedule = new StepDecaySchedule(initialRate: 1f, decayFactor: 0.5f, stepSize: 5);
        var trainer = new Trainer(network, optimizer, schedule);

        // Act
        trainer.Train(inputs, outputs, epochs: 10);

        // Assert - optimizer learning rate should have been adjusted
        // After training completes, the learning rate from the last epoch (epoch 9) is set
        // Epoch 9: 1 * 0.5^(9/5) = 1 * 0.5^1 = 0.5
        Assert.Equal(0.5f, optimizer.LearningRate, precision: 5);
    }
}
