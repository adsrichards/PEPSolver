#include <torch/torch.h>
#include <iostream>

#include "ipeps.h"

int main() {
	int pDim = 2;
	int bDim = 2;
	int cDim = 32;
	int rSteps = 10;

	Ipeps tT(pDim, bDim, cDim, rSteps);

	double x = tT.forward();
}