#pragma once
#include <torch/torch.h>

double tNorm(torch::Tensor tT);
torch::Tensor symmetrize(torch::Tensor tT);