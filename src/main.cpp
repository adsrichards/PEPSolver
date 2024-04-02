#include <torch/torch.h>
#include <iostream>
#include <algorithm>

#include "ipeps.h"
#include "measurement.h"
#include "model.h"

int main() {
	int pDim = 2;
	int bDim = 2;
	int cDim = 16;
	int rSteps = 1;
	int eSteps = 10;

	std::string model_name = "ising";
	double J = 1.0;
	std::tuple<double, double, double> h = { 1.0, 0.0, 0.0 };

	Model model = { model_name, J, h };

	torch::autograd::GradMode::set_enabled(true);

	Ipeps ipeps(pDim, bDim, cDim, rSteps, eSteps, model);
	ipeps.optimize();
}