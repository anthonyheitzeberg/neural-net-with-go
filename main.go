package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// This function initializes a new neural network with the given configuration.
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// Implements the sigmoid activation function.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Implements the derivative of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func main() {
	fmt.Println("Hello, World!")
}
