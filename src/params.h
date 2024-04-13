#pragma once

struct Params
{
    // bond dimensions
    const int pDim = 2;  // physical dim
    const int bDim = 2;  // bond dim
    const int cDim = 10; // corner bond dim
    // iteration steps
    const int rSteps = 1;  // ctmrg
    const int eSteps = 10; // epochs
    // optimization options
    const bool optimize = true;
};