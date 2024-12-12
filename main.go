package main

import (
	"fmt"
	"math/rand"
	"time"
)

type Point struct {
    features []float64
    label    int
}

func generateData(numSamples, numFeatures int) []Point {
	data := make([]Point, numSamples)
	for i := range data {
		data[i].features = make([]float64, numFeatures)
		for j := range data[i].features {
			data[i].features[j] = rand.Float64()
		}
		data[i].label = rand.Intn(2) // Binary classification (0 or 1)
	}
	return data
}

func main() {
	rand.Seed(time.Now().UnixNano())

	numTrainSamples := 10000
	numTestSamples := 200
	numFeatures := 10

	trainData := generateData(numTrainSamples, numFeatures)
	testData := generateData(numTestSamples, numFeatures)

	k := 5 // Number of nearest neighbors

	// Benchmark KNN
	fmt.Println("Benchmarking KNN...")
	durationKNN := benchmarkKNN(trainData, testData, k)
	fmt.Printf("Time taken for KNN classification: %v\n", durationKNN)

	// Benchmark Random Forest
	fmt.Println("Benchmarking Random Forest...")
	durationRF := benchmarkRandomForest(trainData, testData, 100) // Example parameters
	fmt.Printf("Time taken for Random Forest classification: %v\n", durationRF)

	// Benchmark Neural Network
	fmt.Println("Benchmarking Neural Network...")
	durationNN := benchmarkNeuralNetwork(trainData, testData, 32, 100, 0.01) // Example parameters
	fmt.Printf("Time taken for Neural Network training and prediction: %v\n", durationNN)

	// Benchmark PCA
	datasetConfigs := [][2]int{{1000, 50}, {5000, 50}, {10000, 50}}
	threadCounts := []int{1, 2, 4, 8, 16}
	results := runPCABenchmark(datasetConfigs, threadCounts)

	for _, config := range datasetConfigs {
		fmt.Printf("PCA Benchmark for Dataset: %dx%d\n", config[0], config[1])
		fmt.Println("Threads\tTime(s)\tSpeedup")
		for i, threads := range threadCounts {
			fmt.Printf("%d\t%.4f\t%.4f\n", threads,
				results[config]["execution_times"][i],
				results[config]["speedups"][i])
		}
		fmt.Println()
	}
}
