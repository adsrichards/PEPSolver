#include <iostream>
#include <algorithm>

#include "ipeps.h"
#include "measurement.h"
#include "utils.h"

Ipeps::Ipeps(int pDim, int bDim, int cDim, int rSteps, int eSteps, Model model) : 
	Measurement(aT, eT, cT, model), pDim(pDim), bDim(bDim), cDim(cDim), rSteps(rSteps), eSteps(eSteps), model(model){
	//aT = torch::rand({ pDim, bDim, bDim, bDim, bDim });

	aT = torch::ones({ pDim, bDim, bDim, bDim, bDim });
	aT.index_put_({ 0, 0, 0, 0, 0 }, aT.index({ 0, 0, 0, 0, 0 }) + 0.1);

	aT = aT / tNorm(aT);
	aT = register_parameter("A", aT);

	for (auto& param : this->parameters()) {
		param.set_requires_grad(true);
	}

	buildHam(model);
}

Ipeps::~Ipeps() {}

void Ipeps::ctmrg(torch::Tensor& tT) {
	cT = tT.sum(c10::IntArrayRef({ 0, 1 }));
	eT = tT.sum(c10::IntArrayRef({ 1 }));
	eT = eT.permute({ 0, 2, 1 });

	for (int i = 0; i < rSteps; i++) {
		renormalize(tT);
	}
}

void Ipeps::renormalize(torch::Tensor& tT) {
	torch::Tensor rT = rho(tT);
	torch::Tensor pT = std::get<0>(torch::svd(rT));
	const int cDimNew = std::min((int)(eT.size(0) * tT.size(0)), cDim);

	pT = pT.narrow(1, 0, cDimNew);
	renormalizeCorner(rT, pT);

	pT = pT.view({ eT.size(0), tT.size(0), cDimNew });
	renormalizeEdge(tT, pT);
}

torch::Tensor Ipeps::rho(torch::Tensor& tT) {
	torch::Tensor rT = torch::tensordot(cT, eT, { 1 }, { 0 });
	rT = torch::tensordot(rT, eT, { 0 }, { 0 });
	rT = torch::tensordot(rT, tT, { 0, 2 }, { 0, 1 });
	rT = rT.permute({ 0, 3, 1, 2 });

	rT = rT.contiguous().view({ eT.size(0) * tT.size(0), eT.size(0) * tT.size(0) });
	rT = (rT + rT.t());
	rT = rT / tNorm(rT);

	return rT;
}

void Ipeps::renormalizeCorner(torch::Tensor& rT, torch::Tensor& pT) {
	cT = torch::mm(rT, pT);
	cT = torch::mm(pT.t(), cT);
	cT = (cT + cT.t());
	cT = cT / tNorm(cT);
}

void Ipeps::renormalizeEdge(torch::Tensor& tT, torch::Tensor& pT) {
	eT = torch::tensordot(eT, pT, { 0 }, { 0 });
	eT = torch::tensordot(eT, tT, { 0, 2 }, { 1, 0 });
	eT = torch::tensordot(eT, pT, { 0, 2 }, { 0, 1 });
	eT = (eT + eT.permute({ 2, 1, 0 }));
	eT = eT / tNorm(eT);
}

torch::Tensor Ipeps::forward() {
	torch::Tensor bT = symmetrize(aT);

	torch::Tensor tT = torch::mm(bT.view({ pDim, -1 }).t(), bT.view({ pDim, -1 }));
	tT = tT.contiguous().view({ bDim, bDim, bDim, bDim, bDim, bDim, bDim, bDim });

	tT = tT.permute({ 0, 4, 1, 5, 2, 6, 3, 7 });
	tT = tT.contiguous().view({ bDim * bDim, bDim * bDim, bDim * bDim, bDim * bDim });

	tT = tT / tNorm(tT);

	ctmrg(tT);

	return measure();
}

void Ipeps::optimize() {
	torch::optim::LBFGS optimizer(parameters(), torch::optim::LBFGSOptions().max_iter(10));

	auto closure = [&]() -> torch::Tensor {
		optimizer.zero_grad();
		auto loss = forward();
		loss.backward();
		return loss;
		};

	for (int i = 1; i <= eSteps; i++) {
		std::cout << '\n' << "Starting iteration " << i << std::endl;
		auto loss = optimizer.step(closure);

		std::cout << "Completed gradient step" << std::endl;
		std::cout << std::fixed << std::setprecision(12) 
			      << "Loss: " << loss.item<double>() << ", "
		          << "Gradient norm: " << torch::norm(parameters()[0].grad()).item<double>() 
			      << std::endl;

		print_measurements();
	}
}