#include <iostream>

#include "ipeps.h"
#include "utils.h"

Ipeps::Ipeps(int pDim, int bDim, int cDim, int rSteps) : pDim(pDim), bDim(bDim), cDim(cDim), rSteps(rSteps){
	aT = torch::rand({ pDim, bDim, bDim, bDim, bDim });
	aT = aT / torch::norm(aT).item<double>();
}

Ipeps::~Ipeps() {}

inline double tNorm(torch::Tensor tT) {
	return torch::norm(tT).item<double>();
}

double Ipeps::forward() {
	symmetrize(aT);

	torch::Tensor tT = torch::mm(aT.view({ pDim, -1 }).t(), aT.view({ pDim, -1 }));
	tT = tT.contiguous().view({ bDim, bDim, bDim, bDim, bDim, bDim, bDim, bDim });

	tT = tT.permute({ 0, 4, 1, 5, 2, 6, 3, 7 });
	tT = tT.contiguous().view({ bDim * bDim, bDim * bDim, bDim * bDim, bDim * bDim });
	
	tT = tT / tNorm(tT);

	ctmrg(tT);

	return 0.0;
}

void Ipeps::ctmrg(torch::Tensor& tT) {
	torch::Tensor cT = tT.sum(c10::IntArrayRef({ 0, 1 }));
	torch::Tensor eT = tT.sum(c10::IntArrayRef({ 1 }));

	eT = eT.permute({ 0, 2, 1 });

	std::cout << rSteps << std::endl;

	renormalize(tT, cT, eT);

	for (int i = 0; i < rSteps; i++) {
		eT = eT / tNorm(eT);
		renormalize(tT, cT, eT);
	}
}

void Ipeps::renormalize(torch::Tensor& tT, torch::Tensor& cT, torch::Tensor& eT) {
	std::cout << "hello tensors" << std::endl;
}