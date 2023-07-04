#include <sys/time.h>
#include "gtest/gtest.h"
#include "math.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <chrono>
#include "../src/Function.hpp"
#include "../src/Optimizer.hpp"
#include "../src/Function.hpp"
#include "../src/FactoryOptimizer.hpp"


float J2( const std::vector<float>& x ) {
    return pow( x[0] - 1 , 2 ) + pow( x[1] - 2 , 2 );
}

Eigen::MatrixXd DJ2( const std::vector<float>& x ){
    Eigen::MatrixXd M(2,1); 
    M(0,0) = 2*x[0] - 2;
    M(1,0) = 2*x[1] - 4;
    return M;
}

Eigen::MatrixXd H2( const std::vector<float>& x ){
    Eigen::MatrixXd M(2,2); 
    M(0,0) = 2;
    M(1,0) = 0;
    M(1,1) = 2;
    M(0,1) = 0;

    return M;
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



class Function2dTest: public ::testing::Test {
    protected:
        virtual void SetUp() {
            func_  = J2;
            Dfunc_ = DJ2;
            Hfunc_ = H2;
            ann_   = annealing_schedule;
            f_neighbor_ = neighbor;
            f_p_ = p;
        }
        std::vector<float> x0_ = { 0.5 , 1.0 };
        float (*func_)( const::std::vector<float>& x);
        Eigen::MatrixXd (*Dfunc_)( const std::vector<float>& x );
        Eigen::MatrixXd (*Hfunc_)( const std::vector<float>& x );
        float (*ann_)( float );
        std::vector<float> (*f_neighbor_)( const std::vector<float>& );
        float (*f_p_)( float , float , float );

};


TEST_F( Function2dTest , testNewtonRhapson ) {

    auto lambda_cost   = [this](const std::vector<float>& vec) -> float { return func_(vec); };
    auto lambda_Dcost  = [this](const std::vector<float>& vec) -> Eigen::MatrixXd { return Dfunc_(vec); };
    auto lambda_D2cost = [this](const std::vector<float>& vec) -> Eigen::MatrixXd { return Hfunc_(vec); };
    Function cost = Function( lambda_cost , lambda_Dcost , lambda_D2cost , 2 );

    OptimizerParams params = { .type = OptimizerParams::Type::NewtonRaphson,
            .cost_function = cost,
            .x0 = x0_,
            .maxIters = 32 };

    auto rf = FactoryOptimizer::makeOptimizer(params);

    rf->solve();
    std::vector<float> state = rf->get_last_state();
    float err_state = std::abs( state[0]-1 ) + std::abs( state[1]-2 );

    EXPECT_LT( err_state , 1.e-3 );
    EXPECT_LT( rf->get_last_cost() , 1e-3 );
  
}


TEST_F( Function2dTest , testGradientDescent ) {

    auto lambda_cost  = [this](const std::vector<float>& vec) -> float { return func_(vec); };
    auto lambda_Dcost = [this](const std::vector<float>& vec) -> Eigen::MatrixXd { return Dfunc_(vec); };
    auto scale = 0.1;
    Function cost = Function( lambda_cost , lambda_Dcost , 2 );

    OptimizerParams params = { .type = OptimizerParams::Type::GradientDescent,
            .cost_function = cost,
            .x0 = x0_,
            .maxIters = 128,
            .scale = scale};

    auto rf = FactoryOptimizer::makeOptimizer(params);

    rf->solve();
    std::vector<float> state = rf->get_last_state();
    float err_state = std::abs( state[0]-1 ) + std::abs( state[1]-2 );

    EXPECT_LT( err_state , 1.e-3 );
    EXPECT_LT( rf->get_last_cost() , 1e-3 );
  
}


TEST_F( Function2dTest , testGradientDescentMomentum ) {

    auto lambda_cost  = [this](const std::vector<float>& vec) -> float { return func_(vec); };
    auto lambda_Dcost = [this](const std::vector<float>& vec) -> Eigen::MatrixXd { return Dfunc_(vec); };
    auto scale = 0.1;
    Function cost = Function( lambda_cost , lambda_Dcost , 2 );

    OptimizerParams params = { .type = OptimizerParams::Type::GradientDescentMomentum,
            .cost_function = cost,
            .x0 = x0_,
            .maxIters = 128,
            .scale = scale,
            .momentum = 0.1 };

    auto rf = FactoryOptimizer::makeOptimizer(params);

    rf->solve();
    std::vector<float> state = rf->get_last_state();
    float err_state = std::abs( state[0]-1 ) + std::abs( state[1]-2 );

    EXPECT_LT( err_state , 1.e-3 );
    EXPECT_LT( rf->get_last_cost() , 1e-3 );
  
}



TEST_F( Function2dTest , testSimulatedAnnealing ) {

    auto lambda_cost  = [this](const std::vector<float>& vec) -> float { return func_(vec); };
    auto scale = 0.1;
    Function cost = Function( lambda_cost , 2 );

    auto lambda_ann   = [this](float k) -> float { return ann_(k); };
    auto lambda_neigh = [this](const std::vector<float>& vec) -> std::vector<float> { return f_neighbor_(vec); };
    auto lambda_p     = [this](float ex,float enew,float T) -> float { return f_p_(ex,enew,T); };
    SimAnnParams psa;
    psa.f_annealing_       = lambda_ann;
    psa.f_random_neighbor_ = lambda_neigh;
    psa.prob_accept_       = lambda_p;

    OptimizerParams params = { .type = OptimizerParams::Type::SimulatedAnnealing,
            .cost_function = cost,
            .x0 = x0_,
            .maxIters = 256,
            .sim_ann_params = psa };

    auto rf = FactoryOptimizer::makeOptimizer(params);

    rf->solve();
    std::vector<float> state = rf->get_last_state();
    float err_state = std::abs( state[0]-1 ) + std::abs( state[1]-2 );

    EXPECT_LT( err_state , 1.e-1 );
    EXPECT_LT( rf->get_last_cost() , 1e-1 );
  
}
