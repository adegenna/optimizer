#include <math.h>
#include <Eigen/Dense>
#include "../src/Optimizer.hpp"
#include "../src/FactoryOptimizer.hpp"
#include <random>
#include <cmath>
#include <chrono> 

float J( const std::vector<float>& x ) {
    return 0.5 * pow( x[0]-1 , 2 ) + pow( x[1] - 2 , 2 );
}

float annealing_schedule( float k ) {
    float T_0 = 10.0;
    return T_0 / ( k + 1.0 );
}

std::vector<float> neighbor( const std::vector<float>& x ) {
    
    std::default_random_engine generator;
    generator.seed( std::chrono::system_clock::now().time_since_epoch().count() );
    std::vector<float> x_next;
    for ( int i=0; i<x.size(); i++ ) {
        std::normal_distribution<float> distribution( x[i] , 0.1 );
        x_next.push_back( distribution(generator) );
    }
    return x_next;
}

float p( float E_x , float E_xnew , float T ) {
    return exp( -( E_xnew - E_x ) / T );
}


int main() {

    auto lambda_cost  = [](const std::vector<float>& vec) -> float { return J(vec); };
    std::vector<float> x0 = { 0.5 , 1 };
    Function cost( lambda_cost , 2 );

    auto lambda_ann   = [](float k) -> float { return annealing_schedule(k); };
    auto lambda_neigh = [](const std::vector<float>& vec) -> std::vector<float> { return neighbor(vec); };
    auto lambda_p     = [](float ex,float enew,float T) -> float { return p(ex,enew,T); };
    SimAnnParams p;
    p.f_annealing_       = lambda_ann;
    p.f_random_neighbor_ = lambda_neigh;
    p.prob_accept_       = lambda_p;

    OptimizerParams params = { .type = OptimizerParams::Type::SimulatedAnnealing,
            .cost_function = cost,
            .x0 = x0,
            .maxIters = 100,
            .sim_ann_params = p };

    auto rf = FactoryOptimizer::makeOptimizer(params);
    
    rf->solve();
    
    rf->output_solver_history( "rf_history.out" );

}