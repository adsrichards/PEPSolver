#pragma once
#include <torch/torch.h>
#include "model.h"

class Ipeps : public torch::nn::Module
{
public:
	Ipeps(int pDim, int bDim, int cDim, int rSteps, Model model);
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
	double rThresh;

	Model model;

	void ctmrg(torch::Tensor& aT);
	void renormalize(torch::Tensor& aT, torch::Tensor& cT, torch::Tensor& eT);
	torch::Tensor rho(torch::Tensor& tT, torch::Tensor& cT, torch::Tensor& eT);
	void Ipeps::renormalizeEdge(torch::Tensor& eT, torch::Tensor& tT, torch::Tensor& pT);
	void Ipeps::renormalizeCorner(torch::Tensor& cT, torch::Tensor& rT, torch::Tensor& pT);

	torch::Tensor forward();
};

