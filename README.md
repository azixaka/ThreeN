# ThreeN Neural Network Library for .NET

[![Build Status](https://travis-ci.org/ThreeN/ThreeN.svg?branch=master)](https://travis-ci.org/ThreeN/ThreeN)
[![NuGet](https://img.shields.io/nuget/v/ThreeN.svg)](https://www.nuget.org/packages/ThreeN/)

## Introduction

ThreeN is a versatile neural network library for .NET. This library simplifies the process of creating and training neural networks of various configurations. It additionally provides basic matrix operations that are required when working with neural networks.

## Key Features

- Flexibility in creating neural networks of any configuration including:
  - Arbitrary number and size of hidden layers
  - Any number and size of inputs
  - Any number and size of outputs

- Implements training algorithms using:
  - Finite Difference
  - Back Propagation

## Installation

To install ThreeN, you can use the [NuGet package manager](https://www.nuget.org/packages/ThreeN/):

```shell
Install-Package ThreeN
```

## Quick Start

Here is a basic usage example using the fluent API:

```csharp
using ThreeN.LinearAlgebra;
using ThreeN.NeuralNetwork;

var rawData = new float[]
{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 2,
    2, 2, 4,
    3, 2, 5,
    4, 3, 7,
    5, 5, 10
};

// Split into Input Parameters Data
var inputs = new Matrix<float>(rawData.Length / 3, 2, 0, 3, rawData);

// And Expected Output Data
var outputs = new Matrix<float>(rawData.Length / 3, 1, 2, 3, rawData);

// Build network with fluent API: 2 inputs → 5 hidden (ReLU) → 1 output (Linear)
var network = new NeuralNetworkBuilder()
    .WithInputs(2)
    .WithHiddenLayer(5, ActivationFunctionType.Relu)
    .WithOutputLayer(1, ActivationFunctionType.PassThrough)
    .WithInitialization(WeightInitialization.He) // He, Xavier, or Random
    .Build();

// Create gradient structure for training
var gradient = Gradient.CreateFor(network);

// Train the network
for (int epoch = 0; epoch < 10_000; epoch++)
{
    NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
    network.ApplyGradient(gradient, learningRate: 1e-3f);
}

// Use the trained network
inputs.CopyRow(network.InputLayer, rowIndex: 0);
network.Forward();
Console.WriteLine($"Prediction: {network.OutputLayer[0, 0]}");
```

### XOR Example

```csharp
var xorData = new float[]
{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

var inputs = new Matrix<float>(4, 2, 0, 3, xorData);
var outputs = new Matrix<float>(4, 1, 2, 3, xorData);

// Build XOR network: 2 inputs → 2 hidden (Sigmoid) → 1 output (Sigmoid)
var network = new NeuralNetworkBuilder()
    .WithInputs(2)
    .WithHiddenLayer(2, ActivationFunctionType.Sigmoid)
    .WithOutputLayer(1, ActivationFunctionType.Sigmoid)
    .WithInitialization(WeightInitialization.Xavier)
    .Build();

var gradient = Gradient.CreateFor(network);

for (int epoch = 0; epoch < 100_000; epoch++)
{
    NeuralNetworkExtensions.BackPropagation(network, gradient, inputs, outputs);
    network.ApplyGradient(gradient, learningRate: 1f);
}

// Test the network
for (int i = 0; i < inputs.Rows; i++)
{
    inputs.CopyRow(network.InputLayer, i);
    network.Forward();
    Console.WriteLine($"{inputs[i, 0]} XOR {inputs[i, 1]} = {network.OutputLayer[0, 0]:F4}");
}
```

License
This project is licensed under the GPL3 License - see the LICENSE.md file for details.
