#pragma once
#include "torch/torch.h"
#include "model.h"
#include "params.h"
#include "measurement.h"

class Ipeps : public torch::nn::Module, public Measurement
{
public:
	torch::autograd::AutogradContext* ctx;

	Ipeps(Model model, Params params);
	~Ipeps();

	void set_pDim(const int dim);
	int get_pDim() const;

	void set_bDim(const int dim);
	int get_bDim() const;

	void set_cDim(const int dim);
	int get_cDim() const;

	void set_rSteps(const int steps);
	int get_rSteps() const;

	void set_eSteps(const int steps);
	int get_eSteps() const;

	// optimization
	void optimize();

	// tensors
	torch::Tensor aTen;
	torch::Tensor cTen;
	torch::Tensor eTen;

private:
	// initializtion
	enum class Init_type
	{
		init_default,
		init_random,
		init_debug,
	};

	void init_aTen();
	void init_aTen(Init_type init_type);
	void init_aTen_random();
	void init_aTen_offsetOnes();

	void init_cTen(torch::Tensor& tTen);
	void init_eTen(torch::Tensor& tTen);

	// renormalization
	torch::Tensor build_tTen();
	void ctmrg();
	torch::Tensor cTen_absorb_tTen(torch::Tensor& tTen) const;
	void renormalize(torch::Tensor& tTen);
	void renormalize_corner(torch::Tensor& rTen, torch::Tensor& pTen);
	void renormalize_edge(torch::Tensor& rTen, torch::Tensor& pTen, const int new_cDim);

	// optimization
	torch::Tensor forward();

	// model (e.g. "ising")
	Model model;

	// bond dimensions
	int pDim; // physical dim
	int bDim; // bond dim
	int cDim; // corner bond dim
	// iteration steps
	int rSteps; // ctmrg
	int eSteps; // epochs
};

