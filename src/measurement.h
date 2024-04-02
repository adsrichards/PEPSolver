#pragma once
#include <torch/torch.h>
#include <unordered_map>
#include <string>

#include "model.h"

class Measurement
{
public:
	Measurement(torch::Tensor& aT, torch::Tensor& eT, torch::Tensor& cT, Model model);
	torch::Tensor measure();

	void buildHam(Model model);
	void print_measurements();

private:
	std::unordered_map<std::string, double> measurements;

	torch::Tensor& aT;
	torch::Tensor& eT;
	torch::Tensor& cT;

	Model model;

	const torch::Tensor sx = torch::tensor({ {0.0, 1.0}, {1.0, 0.0} });
	const torch::Tensor sy = torch::tensor({ {0.0,-1.0}, {1.0, 0.0} });
	const torch::Tensor sz = torch::tensor({ {1.0, 0.0}, {0.0,-1.0} });
	const torch::Tensor i2 = torch::tensor({ {1.0, 0.0}, {0.0, 1.0} });

	std::tuple<double, double, double> h;
	double J;

	torch::Tensor ham;
};

