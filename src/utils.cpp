#include "utils.h"

void symmetrize(torch::Tensor& tT) {
	tT = (tT + tT.permute({ 0, 1, 4, 3, 2 })) / 2.0;
	tT = (tT + tT.permute({ 0, 3, 2, 1, 4 })) / 2.0;
	tT = (tT + tT.permute({ 0, 4, 3, 2, 1 })) / 2.0;
	tT = (tT + tT.permute({ 0, 2, 1, 4, 3 })) / 2.0;
	tT = tT / torch::norm(tT).item<double>();
}