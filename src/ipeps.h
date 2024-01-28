#pragma once
#include <torch/torch.h>
#include "model.h"
#include "measurement.h"

class Ipeps : public torch::nn::Module, public Measurement
{
public:
	Ipeps(int pDim, int bDim, int cDim, int rSteps, int eSteps, Model model);
	~Ipeps();

	torch::Tensor aT;
	torch::Tensor cT;
	torch::Tensor eT;

	void Ipeps::optimize();
private:
	int pDim;
	int bDim;
	int cDim;
	int rSteps;
	int eSteps;
	double rThresh;

	Model model;

	void ctmrg(torch::Tensor& aT);
	void renormalize(torch::Tensor& tT);
	torch::Tensor rho(torch::Tensor& tT);
	void Ipeps::renormalizeEdge(torch::Tensor& tT, torch::Tensor& pT);
	void Ipeps::renormalizeCorner(torch::Tensor& rT, torch::Tensor& pT);

	torch::Tensor forward();
};

