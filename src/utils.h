#pragma once
#include <torch/torch.h>

double ten_norm(torch::Tensor tT);
torch::Tensor symmetrize_aTen(torch::Tensor tT);