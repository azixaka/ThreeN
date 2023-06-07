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

Here is a basic usage example:

```csharp
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

		// split into Input Parameters Data
        var inData = new Matrix<float>(rawData.Length / 3, 2, 0, 3, rawData);
		
		// and Expected Output Data
        var outData = new Matrix<float>(rawData.Length / 3, 1, 2, 3, rawData);

        var configuration = new[] { 2, 2, 1 }; // 2 inputs, 2 neurons in the hidden layer, 1 output in the output layer
		
		// Each layer can have its own activation function
        var activations = new[] { ActivationFunctionType.Relu, ActivationFunctionType.PassThrough };

        var nn = NeuralNetwork.Create(activations, configuration);
		NeuralNetworkExtensions.HeInitialise(nn); // Supports He, Xavier, Random with low-high
        var gradient = NeuralNetwork.Create(activations, configuration);
		
		for (int i = 0; i < 10_000; i++) // epochs
        {        
            NeuralNetworkExtensions.BackPropagation(nn, gradient, inData, outData);
            NeuralNetworkExtensions.Train(nn, gradient, 1e-3f); // learning rate
        }
```

License
This project is licensed under the GPL3 License - see the LICENSE.md file for details.
