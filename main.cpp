#include <torch/torch.h>
#include <iostream>

#include "ipeps.h"

void printRandTensor() {
	torch::Tensor tensor = torch::rand({ 2, 3 });
	std::cout << tensor << std::endl;
}

int main() {
	Ipeps T(1,2,3,4);
}