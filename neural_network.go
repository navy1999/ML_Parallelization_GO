package main

import (
	"math"
	"math/rand"
	"sync"
)

type NeuralNetwork struct {
	inputSize  int
	hiddenSize int
	outputSize int
	weights1   [][]float64
	weights2   [][]float64
	bias1      []float64
	bias2      []float64
}

func NewNeuralNetwork(inputSize, hiddenSize, outputSize int) *NeuralNetwork {
	nn := &NeuralNetwork{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		weights1:   make([][]float64, hiddenSize),
		weights2:   make([][]float64, outputSize),
		bias1:      make([]float64, hiddenSize),
		bias2:      make([]float64, outputSize),
	}

	for i := range nn.weights1 {
		nn.weights1[i] = make([]float64, inputSize)
		for j := range nn.weights1[i] {
			nn.weights1[i][j] = rand.Float64()*2 - 1
		}
	}

	for i := range nn.weights2 {
		nn.weights2[i] = make([]float64, hiddenSize)
		for j := range nn.weights2[i] {
			nn.weights2[i][j] = rand.Float64()*2 - 1
		}
	}

	for i := range nn.bias1 {
		nn.bias1[i] = rand.Float64()*2 - 1
	}

	for i := range nn.bias2 {
		nn.bias2[i] = rand.Float64()*2 - 1
	}

	return nn
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (nn *NeuralNetwork) Forward(input []float64) []float64 {
	hidden := make([]float64, nn.hiddenSize)
	output := make([]float64, nn.outputSize)

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range hidden {
			sum := 0.0
			for j, val := range input {
				sum += nn.weights1[i][j] * val
			}
			hidden[i] = sigmoid(sum + nn.bias1[i])
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range output {
			sum := 0.0
			for j, val := range hidden {
				sum += nn.weights2[i][j] * val
			}
			output[i] = sigmoid(sum + nn.bias2[i])
		}
	}()

	wg.Wait()

	return output
}

func (nn *NeuralNetwork) Train(inputs [][]float64, targets [][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i, input := range inputs {
			// Forward pass
			hidden := make([]float64, nn.hiddenSize)
			output := make([]float64, nn.outputSize)

			for j := range hidden {
				sum := 0.0
				for k, val := range input {
					sum += nn.weights1[j][k] * val
				}
				hidden[j] = sigmoid(sum + nn.bias1[j])
			}

			for j := range output {
				sum := 0.0
				for k, val := range hidden {
					sum += nn.weights2[j][k] * val
				}
				output[j] = sigmoid(sum + nn.bias2[j])
			}

			// Backpropagation
			outputErrors := make([]float64, nn.outputSize)
			for j := range outputErrors {
				outputErrors[j] = targets[i][j] - output[j]
			}

			hiddenErrors := make([]float64, nn.hiddenSize)
			for j := range hiddenErrors {
				sum := 0.0
				for k, val := range outputErrors {
					sum += val * nn.weights2[k][j]
				}
				hiddenErrors[j] = sum
			}

			// Update weights and biases
			var wg sync.WaitGroup

			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range nn.weights2 {
					for k := range nn.weights2[j] {
						nn.weights2[j][k] += learningRate * outputErrors[j] * output[j] * (1 - output[j]) * hidden[k]
					}
					nn.bias2[j] += learningRate * outputErrors[j] * output[j] * (1 - output[j])
				}
			}()

			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range nn.weights1 {
					for k := range nn.weights1[j] {
						nn.weights1[j][k] += learningRate * hiddenErrors[j] * hidden[j] * (1 - hidden[j]) * input[k]
					}
					nn.bias1[j] += learningRate * hiddenErrors[j] * hidden[j] * (1 - hidden[j])
				}
			}()

			wg.Wait()
		}
	}
}
