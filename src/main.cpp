#include <torch/torch.h>
#include <iostream>

#include "ipeps.h"

void printRandTensor() {
	torch::Tensor tensor = torch::rand({ 2, 3 });
	std::cout << tensor << std::endl;
}

int main() {
	int pDim = 2;
	int bDim = 2;
	int cDim = 8;
	int rSteps = 10;

	Ipeps tT(pDim, bDim, cDim, rSteps);

	double x = tT.forward();
}