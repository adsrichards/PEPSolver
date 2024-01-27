#include "utils.h"

void symmetrize(torch::Tensor& A) {
	A = (A + A.permute({ 0, 1, 4, 3, 2 })) / 2.0;
	A = (A + A.permute({ 0, 3, 2, 1, 4 })) / 2.0;
	A = (A + A.permute({ 0, 4, 3, 2, 1 })) / 2.0;
	A = (A + A.permute({ 0, 2, 1, 4, 3 })) / 2.0;
	A = A / torch::norm(A).item<double>();
}