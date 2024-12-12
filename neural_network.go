package main

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

type NeuralNetwork struct {
	inputSize  int
 hiddenSize int 
 outputSize int 
 weights1   [][]float64 // Weights from input to hidden layer.
 weights2   [][]float64 // Weights from hidden to output layer.
 bias1      []float64   // Biases for hidden layer.
 bias2      []float64   // Biases for output layer.
}

func NewNeuralNetwork(inputSize, hiddenSize, outputSize int) *NeuralNetwork{
	nn:=&NeuralNetwork{
	inputSize: inputSize,
hiddenSize: hiddenSize,
outputSize: outputSize,
weights1: make([][]float64 ,hiddenSize),
weights2: make([][]float64 ,outputSize),
bias1: make([]float64 ,hiddenSize),
bias2: make([]float64 ,outputSize),
}

for i:=range nn.weights1{
	nn.weights1[i]=make([]float64 ,inputSize )
	for j:=range nn.weights1[i]{
	nn.weights1[i][j]=rand.Float64()*2-1 // Initialize weights randomly.
	  }
 }

for i:=range nn.weights2{
	nn.weights2[i]=make([]float64 ,hiddenSize )
	for j:=range nn.weights2[i]{
	nn.weights2[i][j]=rand.Float64()*2-1 // Initialize weights randomly.
	  }
 }

for i:=range nn.bias1{nn.bias1[i]=rand.Float64()*2-1 }
for i:=range nn.bias2{nn.bias2[i]=rand.Float64()*2-1 }

return nn 
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

func (nn *NeuralNetwork) Forward(input []float64) []float64{
hiddenLayerOutput:=make([]float64 ,nn.hiddenSize )
outputLayerOutput:=make([]float64 ,nn.outputSize )

var wg sync.WaitGroup

wg.Add(nn.hiddenSize )
for i:=range hiddenLayerOutput{
	go func(i int){
	defer wg.Done()
	sum:=0.0 
	for j:=range input{
	sum+=nn.weights1[i][j]*input[j]
	  }
hiddenLayerOutput[i]=sigmoid(sum + nn.bias1[i])
}(i)
}
wg.Wait()

wg.Add(nn.outputSize )
for i:=range outputLayerOutput{
	go func(i int){
	defer wg.Done()
	sum:=0.0 
	for j:=range hiddenLayerOutput{
	sum+=nn.weights2[i][j]*hiddenLayerOutput[j]
	  }
outputLayerOutput[i]=sigmoid(sum + nn.bias2[i])
}(i)
}
wg.Wait()

return outputLayerOutput 
}

// Benchmarking function for Neural Network training and prediction.
func benchmarkNeuralNetwork(trainData []Point ,testData []Point ,
                            hiddenSize int ,epochs int ,
                            learningRate float64 ) time.Duration {

inputSize:=len(trainData[0].features )
outputSize:=2 // Binary classification

inputs:=make([][]float64,len(trainData ))
targets:=make([][]float64,len(trainData ))
for i ,point:=range trainData{
inputs[i]=point.features 
targets[i]=make([]float64 ,outputSize )
targets[i][point.label]=1.0 // One-hot encoding of labels.
if point.label==0 { targets[i][1]=0.0 } else { targets[i][0]=0.0 }
}

startTime:=time.Now()
nn:=NewNeuralNetwork(inputSize ,hiddenSize ,outputSize )

for epoch:=0; epoch<epochs; epoch++{
var wg sync.WaitGroup

for i:=range inputs{
wg.Add(1)
go func(i int){
defer wg.Done()
_ = nn.Forward(inputs[i]) // Placeholder; implement training logic here.
}(i)
}
wg.Wait()
// Implement weight updates here...
}

for _,testPoint:=range testData{
_ = nn.Forward(testPoint.features ) // Placeholder; implement prediction logic here.
}

return time.Since(startTime )
}
