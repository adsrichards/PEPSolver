#include "ipeps.h"
#include "torch/torch.h"
#include "measurement.h"
#include "model.h"
#include "params.h"
#include "utils.h"

#include <iostream>
#include <algorithm>

#define DEBUG
#ifdef DEBUG
#define LOG(x) std::cout << x << std::endl;
#endif

Ipeps::Ipeps(Model model, Params params)
    : Measurement(aTen, eTen, cTen, model),
      bDim(params.bDim), pDim(params.pDim), cDim(params.cDim),
      rSteps(params.rSteps), eSteps(params.eSteps),
      model(model)
{
    init_aTen();

    if (params.optimize)
    {
        torch::autograd::GradMode::set_enabled(true);
        aTen = register_parameter("A", aTen);
        for (auto &aTen_param : this->parameters())
            aTen_param.set_requires_grad(true);
    }

    buildHam(model);
}

Ipeps::~Ipeps() {}

void Ipeps::set_pDim(const int dim) { pDim = dim; }
int Ipeps::get_pDim() const { return pDim; };

void Ipeps::set_bDim(const int dim) { bDim = dim; }
int Ipeps::get_bDim() const { return bDim; };

void Ipeps::set_cDim(const int dim) { cDim = dim; }
int Ipeps::get_cDim() const { return cDim; };

void Ipeps::set_rSteps(const int steps) { cDim = steps; }
int Ipeps::get_rSteps() const { return rSteps; };

void Ipeps::set_eSteps(const int steps) { eSteps = steps; }
int Ipeps::get_eSteps() const { return eSteps; };

// initialization

void Ipeps::init_aTen()
{
    init_aTen(Init_type::init_default);
}

void Ipeps::init_aTen(Init_type init_type)
{
    switch (init_type)
    {
    case Init_type::init_default:
#ifdef DEBUG
        init_aTen(Init_type::init_debug);
        break;
#endif DEBUG
        init_aTen(Init_type::init_random);
        break;
    case Init_type::init_random:
        init_aTen_random();
        break;
    case Init_type::init_debug:
        init_aTen_offsetOnes();
        break;
    }
}

// renormalization

void Ipeps::init_aTen_random()
{
    c10::IntArrayRef aTen_dims = c10::IntArrayRef({pDim, bDim, bDim, bDim, bDim});
    aTen = torch::rand(aTen_dims);
    aTen /= ten_norm(aTen);
}

void Ipeps::init_aTen_offsetOnes()
{
    c10::IntArrayRef aTen_dims = c10::IntArrayRef({pDim, bDim, bDim, bDim, bDim});
    aTen = torch::ones(aTen_dims);
    aTen.index_put_({0, 0, 0, 0, 0}, aTen.index({0, 0, 0, 0, 0}) + 0.1);
    aTen /= ten_norm(aTen);
}

void Ipeps::init_cTen(torch::Tensor &tTen)
{
    cTen = tTen.sum(c10::IntArrayRef({0, 1}));
}

void Ipeps::init_eTen(torch::Tensor &tTen)
{
    eTen = tTen.sum(c10::IntArrayRef({1}));
    eTen = eTen.permute({0, 2, 1});
}

// renormalization

torch::Tensor Ipeps::build_tTen()
{
    torch::Tensor bT = symmetrize_aTen(aTen);
    torch::Tensor tTen = torch::mm(bT.view({pDim, -1}).t(), bT.view({pDim, -1}));
    c10::IntArrayRef tTen_dims = c10::IntArrayRef({bDim, bDim, bDim, bDim, bDim, bDim, bDim, bDim});
    tTen = tTen.contiguous().view(tTen_dims);

    tTen = tTen.permute({0, 4, 1, 5, 2, 6, 3, 7});
    tTen_dims = c10::IntArrayRef({bDim * bDim, bDim * bDim, bDim * bDim, bDim * bDim});
    tTen = tTen.contiguous().view(tTen_dims);

    tTen /= ten_norm(tTen);

    return tTen;
}

void Ipeps::ctmrg()
{
    torch::Tensor tTen = build_tTen();
    init_cTen(tTen);
    init_eTen(tTen);

    for (int i = 0; i < rSteps; ++i)
    {
        renormalize(tTen);
    }
}

void Ipeps::renormalize(torch::Tensor &tTen)
{
    // insertion and absorption step
    torch::Tensor new_cTen = cTen_absorb_tTen(tTen);

    // calculate and truncate/decimate projector
    torch::Tensor pTen = std::get<0>(torch::svd(new_cTen)); // projector
    const int new_cDim = std::min((int)(eTen.size(0) * tTen.size(0)), cDim);
    pTen = pTen.narrow(1, 0, new_cDim);

    // renormalize corner and edge
    renormalize_corner(new_cTen, pTen);
    renormalize_edge(tTen, pTen, new_cDim);
}

torch::Tensor Ipeps::cTen_absorb_tTen(torch::Tensor &tTen) const
{
    torch::Tensor new_cTen = torch::tensordot(cTen, eTen, {1}, {0});
    new_cTen = torch::tensordot(new_cTen, eTen, {0}, {0});
    new_cTen = torch::tensordot(new_cTen, tTen, {0, 2}, {0, 1});
    new_cTen = new_cTen.permute({0, 3, 1, 2});

    c10::IntArrayRef new_cTen_dims = c10::IntArrayRef({eTen.size(0) * tTen.size(0), eTen.size(0) * tTen.size(0)});
    new_cTen = new_cTen.contiguous().view(new_cTen_dims);
    new_cTen = (new_cTen + new_cTen.t());
    new_cTen = new_cTen / ten_norm(new_cTen);

    return new_cTen;
}

void Ipeps::renormalize_corner(torch::Tensor &new_cTen, torch::Tensor &pTen)
{
    cTen = torch::mm(new_cTen, pTen);
    cTen = torch::mm(pTen.t(), cTen);
    cTen = (cTen + cTen.t());
    cTen /= ten_norm(cTen);
}

void Ipeps::renormalize_edge(torch::Tensor &tTen, torch::Tensor &pTen, const int new_cDim)
{
    pTen = pTen.view({eTen.size(0), tTen.size(0), new_cDim});
    eTen = torch::tensordot(eTen, pTen, {0}, {0});
    eTen = torch::tensordot(eTen, tTen, {0, 2}, {1, 0});
    eTen = torch::tensordot(eTen, pTen, {0, 2}, {0, 1});
    eTen = (eTen + eTen.permute({2, 1, 0}));
    eTen = eTen / ten_norm(eTen);
}

// optimization

// forward pass in optimization step
torch::Tensor Ipeps::forward()
{
    ctmrg();
    return measure();
}

void Ipeps::optimize()
{
    torch::optim::LBFGS optimizer(parameters(), torch::optim::LBFGSOptions().max_iter(10));

    auto closure = [&]() -> torch::Tensor
    {
        optimizer.zero_grad();
        auto loss = forward();
        loss.backward();
        return loss;
    };

    for (int i = 1; i <= eSteps; ++i)
    {
        std::cout << '\n'
                  << "Starting iteration " << i << std::endl;
        auto loss = optimizer.step(closure);

        std::cout << "Completed gradient step" << std::endl;
        std::cout << std::fixed << std::setprecision(12)
                  << "Loss: " << loss.item<double>() << ", "
                  << "Gradient norm: " << torch::norm(parameters()[0].grad()).item<double>()
                  << std::endl;

        print_measurements();
    }
}