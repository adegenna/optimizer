#pragma once

#include "Optimizer.hpp"
#include "Parameters.hpp"
#include <iostream>
#include <fstream>
#include <unordered_map>

struct OptimizerParams {
    enum class Type { NewtonRaphson, GradientDescent , GradientDescentMomentum , SimulatedAnnealing , ParticleSwarm };
    Type type;
    Function cost_function;
    NewtRhapParams nr_params;
    GradDesParams gd_params;
    GradDesMomParams gdm_params;
    SimAnnParams sim_ann_params;
    PartSwarmParams part_swarm_params;
};


class FactoryOptimizer {
    public:

        static std::unique_ptr<Optimizer> makeOptimizer( const OptimizerParams& params);
};
