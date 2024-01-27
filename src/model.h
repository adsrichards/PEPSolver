#pragma once
#include <string>
#include <tuple>

struct Model {
	std::string model_name;
	double J;
	std::tuple<double, double, double> h;
	Model(std::string model_name, double J, std::tuple<double, double, double> h);
};