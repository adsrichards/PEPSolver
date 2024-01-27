#include <iostream>

#include "ipeps.h"
#include "utils.h"

Ipeps::Ipeps(int pDim, int bDim, int cDim, int rsteps) : pDim(pDim), bDim(bDim), cDim(cDim), rSteps(rSteps){
	A = torch::rand({ pDim, bDim, bDim, bDim, bDim });
	A = A / torch::norm(A).item<double>();
}

Ipeps::~Ipeps() {}

double Ipeps::forward() {
	symmetrize(A);

	torch::Tensor T = torch::mm(A.view({ pDim, -1 }).t(), A.view({ pDim, -1 }));
	T = T.contiguous().view({ bDim, bDim, bDim, bDim, bDim, bDim, bDim, bDim });

	T = T.permute({ 0, 4, 1, 5, 2, 6, 3, 7 });
	T = T.contiguous().view({ bDim * bDim, bDim * bDim, bDim * bDim, bDim * bDim });
	
	T = T / torch::norm(T).item<double>();
	return 0.0;
}