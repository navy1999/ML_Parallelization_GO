package main

import (
	"math/rand"
)

type Point struct {
	features []float64
	label    int
}

// Function to generate random data points
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
