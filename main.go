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
	    for j:=range data[i].features{
	        data[i].features[j]=rand.Float64()
	    }
	    data[i].label=rand.Intn(2) // Binary classification (label can be either 0 or 1).
	  }
	  return data	
}

func main() {

	rand.Seed(time.Now().UnixNano())

	numTrainSamples:=10000 
	numTestSamples:=200 
	numFeatures:=10 

   trainData:=generateData(numTrainSamples,numFeatures)
   testData:=generateData(numTestSamples,numFeatures)

	k:=5 // Number of nearest neighbors

	fmt.Println("Benchmarking KNN...")
	durationKNN:=benchmarkKNN(trainData,testData,k)
	fmt.Printf("Time taken for KNN classification: %v\n",durationKNN)


	fmt.Println("Benchmarking Random Forest...")
	durationRF:=benchmarkRandomForest(trainData,testData ,100 ) 
	fmt.Printf("Time taken for Random Forest classification: %v\n",durationRF)


	fmt.Println("Benchmarking Neural Network...")
	durationNN:=benchmarkNeuralNetwork(trainData,testData ,32 ,100 ,0.01 )
	fmt.Printf("Time taken for Neural Network training and prediction: %v\n",durationNN )
}
