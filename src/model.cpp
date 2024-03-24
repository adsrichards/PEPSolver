#include "model.h"
#include <assert.h>
#include <algorithm>

std::string model_list[] = {
	"ising",
};

Model::Model(
	std::string model_name,
	double J,
	std::tuple<double, double, double> h) :
	model_name(model_name), J(J), h(h) {
	assert(std::find(std::begin(model_list), std::end(model_list), model_name) != std::end(model_list));
}