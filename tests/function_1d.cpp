#include <sys/time.h>
#include "gtest/gtest.h"
#include "math.h"
#include <Eigen/Dense>
#include "../src/Function.hpp"
#include "../src/Optimizer.hpp"
#include "../src/Function.hpp"
#include "../src/FactoryOptimizer.hpp"


float J( const std::vector<float>& x ) {
    return pow( x[0]-2,2 );
}

Eigen::MatrixXd DJ( const std::vector<float>& x ){
    Eigen::MatrixXd M(1,1); 
    M(0,0) = 2*x[0] - 4;
    return M;
}

Eigen::MatrixXd D2J( const std::vector<float>& x ){
    Eigen::MatrixXd M(1,1); 
    M(0,0) = 2;
    return M;
}


class Function1dTest: public ::testing::Test {
    protected:
        virtual void SetUp() {
            func_   = J;
            Dfunc_  = DJ;
            D2func_ = D2J;
            x0_.push_back( 1 );
        }
        std::vector<float> x0_;
        float (*func_)( const::std::vector<float>& x);
        Eigen::MatrixXd (*Dfunc_)( const std::vector<float>& x );
        Eigen::MatrixXd (*D2func_)( const std::vector<float>& x );

};


TEST_F( Function1dTest , testNewtonRhapson ) {

    auto lambda_cost   = [this](const std::vector<float>& vec) -> float { return func_(vec); };
    auto lambda_Dcost  = [this](const std::vector<float>& vec) -> Eigen::MatrixXd { return Dfunc_(vec); };
    auto lambda_D2cost = [this](const std::vector<float>& vec) -> Eigen::MatrixXd { return D2func_(vec); };
    Function cost = Function( lambda_cost , lambda_Dcost , lambda_D2cost , 1 );

    OptimizerParams params = { .type = OptimizerParams::Type::NewtonRaphson,
                                .cost_function = cost,
                                .x0 = x0_,
                                .maxIters = 8 };

    auto rf = FactoryOptimizer::makeOptimizer(params);

    rf->solve();
    std::vector<float> state = rf->get_last_state();

    EXPECT_LT( std::abs( state[0] - 2 ) , 1.e-4 );
    EXPECT_LT( rf->get_last_cost() , 1e-4 );
  
}
