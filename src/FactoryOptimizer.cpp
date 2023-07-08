#include "FactoryOptimizer.hpp"


std::unique_ptr<Optimizer> FactoryOptimizer::makeOptimizer( const OptimizerParams& params)
{
    switch (params.type) {
        case OptimizerParams::Type::NewtonRaphson:
            return std::make_unique<NewtonRhapson>(params.cost_function, params.x0, params.maxIters);
        case OptimizerParams::Type::GradientDescent:
            return std::make_unique<GradientDescent>(params.cost_function, params.x0, params.maxIters, params.scale);
        case OptimizerParams::Type::GradientDescentMomentum:
            return std::make_unique<GradientDescentMomentum>(params.cost_function, params.x0, params.maxIters, params.scale, params.momentum);
        case OptimizerParams::Type::SimulatedAnnealing:
            return std::make_unique<SimulatedAnnealing>(params.cost_function, params.x0, params.maxIters, params.sim_ann_params);
        case OptimizerParams::Type::ParticleSwarm:
            return std::make_unique<ParticleSwarm>(params.cost_function, params.maxIters, params.part_swarm_params);

    }
    return nullptr;
}