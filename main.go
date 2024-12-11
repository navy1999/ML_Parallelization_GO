package main

import (
	"fmt"
	"math/rand"
	"time"
)

func generateData(numSamples, numFeatures int) []Point {
	data := make([]Point, numSamples)
	for i := range data {
		data[i].features = make([]float64, numFeatures)
		for j := range data[i].features {
			data[i].features[j] = rand.Float64()
		}
		data[i].label = rand.Intn(2)
	}
	return data
}

func benchmarkKNN(trainData, testData []Point, k, numGoroutines int) time.Duration {
	start := time.Now()
	for _, testPoint := range testData {
		knnClassify(trainData, testPoint, k, numGoroutines)
	}
	return time.Since(start)
}

func benchmarkRandomForest(trainData, testData []Point, numTrees, maxDepth, minSamples int) time.Duration {
	start := time.Now()
	rf := &RandomForest{}
	rf.Train(trainData, numTrees, maxDepth, minSamples)
	for _, testPoint := range testData {
		rf.Predict(testPoint)
	}
	return time.Since(start)
}

func benchmarkNeuralNetwork(trainData, testData []Point, hiddenSize, epochs int, learningRate float64) time.Duration {
	inputSize := len(trainData[0].features)
	outputSize := 2 // Binary classification

	inputs := make([][]float64, len(trainData))
	targets := make([][]float64, len(trainData))
	for i, point := range trainData {
		inputs[i] = point.features
		targets[i] = make([]float64, outputSize)
		targets[i][point.label] = 1
	}

	start := time.Now()
	nn := NewNeuralNetwork(inputSize, hiddenSize, outputSize)
	nn.Train(inputs, targets, epochs, learningRate)
	for _, testPoint := range testData {
		nn.Forward(testPoint.features)
	}
	return time.Since(start)
}

func main() {
    rand.Seed(time.Now().UnixNano())

    numTrainSamples := 10000
    numTestSamples := 1000
    numFeatures := 10

    trainData := generateData(numTrainSamples, numFeatures)
    testData := generateData(numTestSamples, numFeatures)

    fmt.Println("KNN Benchmark:")
    knnTime := benchmarkKNN(trainData, testData, 5, 4)
    fmt.Printf("Time taken: %v\n", knnTime)

    fmt.Println("\nRandom Forest Benchmark:")
    rfTime := benchmarkRandomForest(trainData, testData, 100, 10, 5)
    fmt.Printf("Time taken: %v\n", rfTime)

    fmt.Println("\nNeural Network Benchmark:")
    nnTime := benchmarkNeuralNetwork(trainData, testData, 64, 100, 0.01)
    fmt.Printf("Time taken: %v\n", nnTime)
}

