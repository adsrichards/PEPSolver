#pragma once
#include <torch/torch.h>

class Ipeps
{
public:
	Ipeps(int pDim, int bDim, int cDim, int rSteps);
	~Ipeps();
private:
	int pDim;
	int bDim;
	int cDim;
	int rSteps;

	torch::Tensor A;

	double forward();
};

