#pragma once

#include "Optimizer.hpp"
#include "SimAnnParams.hpp"
#include <iostream>
#include <fstream>
#include <unordered_map>

struct OptimizerParams {
    enum class Type { NewtonRaphson, GradientDescent , GradientDescentMomentum , SimulatedAnnealing };
    Type type;
    Function cost_function;
    std::vector<float> x0;
    int maxIters;
    float scale;
    float momentum;
    SimAnnParams sim_ann_params;
};


class FactoryOptimizer {
    public:

        static std::unique_ptr<Optimizer> makeOptimizer( const OptimizerParams& params);
};
