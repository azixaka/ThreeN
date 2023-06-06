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
		var rowData = new float[]
        {
            0, 0, 0,
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        };

        var inData = new Matrix<float>(4, 2, 0, 3, rowData);
        var outData = new Matrix<float>(4, 1, 2, 3, rowData);
        
        var configuration = new[] { 2, 10, 10, 10, 1 };
        var nn = NeuralNetwork.Create(configuration);
        NeuralNetworkExtensions.Randomise(nn, 0, 1);

        TryAllData(inData, nn);

        var cost = NeuralNetworkExtensions.Cost(nn, inData, outData);
        Console.WriteLine($"Pre-training cost: {cost}");

        var nng = NeuralNetwork.Create(configuration);

        var sw = Stopwatch.StartNew();

        for (int i = 0; i < 10_000; i++)
        {         
            NeuralNetworkExtensions.BackPropagation(nn, nng, inData, outData);
            NeuralNetworkExtensions.Train(nn, nng, 1f);
        }

        sw.Stop();
        cost = NeuralNetworkExtensions.Cost(nn, inData, outData);
        Console.WriteLine($"Post-training cost: {cost}; Elapsed: {sw.Elapsed.TotalMilliseconds}ms");

        TryAllData(inData, nn);

        static void TryAllData(Matrix<float> inData, NeuralNetwork nn)
        {
            for (int i = 0; i < inData.Rows; i++)
            {
                inData.CopyRow(nn.InputLayer, i);

                NeuralNetworkExtensions.Forward(nn);

                for (int j = 0; j < inData.Columns; j++)
                {
                    Console.Write($"{inData.ElementAt(i, j)} ");
                }

                Console.WriteLine($"{nn.OutputLayer.ElementAt(0, 0)}");
            }
        }
```

License
This project is licensed under the GPL3 License - see the LICENSE.md file for details.
