#include "utils.h"

double ten_norm(torch::Tensor tT) {
	return torch::norm(tT).item<double>();
}

torch::Tensor symmetrize_aTen(torch::Tensor tT) {
    tT = (tT + tT.permute({ 0, 1, 4, 3, 2 })) / 2.0;
    tT = (tT + tT.permute({ 0, 3, 2, 1, 4 })) / 2.0;
    tT = (tT + tT.permute({ 0, 4, 3, 2, 1 })) / 2.0;
    tT = (tT + tT.permute({ 0, 2, 1, 4, 3 })) / 2.0;
    tT = tT / ten_norm(tT);

    return tT;
}