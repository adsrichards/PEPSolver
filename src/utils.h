#pragma once
#include <torch/torch.h>

double tNorm(torch::Tensor tT);
void symmetrize(torch::Tensor& tT);