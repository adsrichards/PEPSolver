#include "utils.h"

double tNorm(torch::Tensor tT) {
	return torch::norm(tT).item<double>();
}

torch::Tensor symmetrize(torch::Tensor tT) {
    tT = (tT + tT.permute({ 0, 1, 4, 3, 2 })) / 2.0;
    tT = (tT + tT.permute({ 0, 3, 2, 1, 4 })) / 2.0;
    tT = (tT + tT.permute({ 0, 4, 3, 2, 1 })) / 2.0;
    tT = (tT + tT.permute({ 0, 2, 1, 4, 3 })) / 2.0;
    tT = tT / tNorm(tT);

    return tT;
}