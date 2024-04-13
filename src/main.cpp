#include "torch/torch.h"
#include "ipeps.h"
#include "measurement.h"
#include "model.h"
#include "params.h"

int main() {
	// define model
	std::string model_name = "ising";
	double J = 1.0;
	std::tuple<double, double, double> h = { 1.0, 0.0, 0.0 };
	Model model = { model_name, J, h };

	// create iPEPS
	Params params;
	Ipeps ipeps(model, params);
	
	// optimize iPEPS
	ipeps.optimize();
}