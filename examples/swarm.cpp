#include <math.h>
#include <Eigen/Dense>
#include "../src/Optimizer.hpp"
#include "../src/FactoryOptimizer.hpp"
#include <random>
#include <cmath>
#include <chrono> 


float J( const std::vector<float>& x ) {
    return 1.0 - 
            exp( -( pow( x[0]-0.3 , 2 ) + pow( x[1]-0.3 , 2 ) ) / pow(0.2,2) ) -
            0.5 * exp( -( pow( x[0]-0.7 , 2 ) + pow( x[1]-0.7 , 2 ) ) / pow(0.2,2) );
}

std::vector<float> draw_random_state() {
    std::default_random_engine generator;
    generator.seed( std::chrono::system_clock::now().time_since_epoch().count() );
    std::vector<float> x(2);
    std::uniform_real_distribution<float> distribution( 0 , 1 );
    x[0] = distribution( generator );
    x[1] = distribution( generator );
    return x;
}

std::vector<float> draw_random_vel() {
    std::default_random_engine generator;
    generator.seed( std::chrono::system_clock::now().time_since_epoch().count() );
    std::vector<float> v(2);
    std::uniform_real_distribution<float> distribution( -1 , 1 );
    v[0] = distribution( generator );
    v[1] = distribution( generator );
    return v;
}

bool is_done( const std::vector<std::vector<float>>& x ) {
    return false;
}


int main() {

    auto lambda_cost  = [](const std::vector<float>& vec) -> float { return J(vec); };
    Function cost( lambda_cost , 2 );

    auto lambda_randState = []() -> std::vector<float> { return draw_random_state(); };
    auto lambda_randVel   = []() -> std::vector<float> { return draw_random_vel(); };
    auto lambda_isDone    = [](const std::vector<std::vector<float>>& x) -> bool { return is_done(x); };

    PartSwarmParams p;
    p.n_swarm_ = 16;
    p.is_done_ = lambda_isDone;
    p.draw_random_state_ = lambda_randState;
    p.draw_random_vel_   = lambda_randVel;
    p.w_     = 0.33;
    p.phi_p_ = 0.33;
    p.phi_g_ = 0.33;

    OptimizerParams params = { .type = OptimizerParams::Type::ParticleSwarm,
            .cost_function = cost,
            .maxIters = 200,
            .part_swarm_params = p };

    auto rf = FactoryOptimizer::makeOptimizer(params);
    
    rf->solve();
    
    rf->output_solver_history( "rf_history.out" );

}