#pragma once
#include <torch/torch.h>

class Ipeps
{
public:
	Ipeps(int pDim, int bDim, int cDim, int rSteps);
	~Ipeps();

	double forward();
private:
	int pDim;
	int bDim;
	int cDim;
	int rSteps;
	double rThresh;

	torch::Tensor aT;

	void ctmrg(torch::Tensor& aT);
	void renormalize(torch::Tensor& aT, torch::Tensor& cT, torch::Tensor& eT);
};

