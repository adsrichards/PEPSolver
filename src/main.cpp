#include <torch/torch.h>
#include <iostream>
#include <algorithm>

#include "ipeps.h"
#include "measurement.h"
#include "model.h"

int main() {
	int pDim = 2;
	int bDim = 2;
	int cDim = 32;
	int rSteps = 10;

	std::string model_list;
	std::string model_name = "ising";
	double J = 1.0;
	std::tuple<double, double, double> h = { 1.0, 0.0, 0.0 };

	Model model = { model_name, J, h };

	Ipeps ipeps(pDim, bDim, cDim, rSteps);
	double x = ipeps.forward();

	Measurement m(ipeps.aT, ipeps.eT, ipeps.cT, model);
	auto measurements = m.measure();
}