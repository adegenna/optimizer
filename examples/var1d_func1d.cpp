#include <math.h>
#include <Eigen/Dense>
#include "../src/Optimizer.hpp"
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


int main() {

    std::vector<float> x0 = { 1 };
    auto lambda_cost   = [](const std::vector<float>& vec) -> float { return J(vec); };
    auto lambda_Dcost  = [](const std::vector<float>& vec) -> Eigen::MatrixXd { return DJ(vec); };
    auto lambda_D2cost = [](const std::vector<float>& vec) -> Eigen::MatrixXd { return D2J(vec); };
    Function cost = Function( lambda_cost , lambda_Dcost , lambda_D2cost , 1 );

    NewtRhapParams p;
    p.maxIters_ = 8;
    p.x0_       = x0;

    OptimizerParams params = { .type = OptimizerParams::Type::NewtonRaphson,
                                .cost_function = cost,
                                .nr_params     = p };

    auto rf = FactoryOptimizer::makeOptimizer(params);

    rf->solve();
    rf->output_solver_history( "rf_history.out" );

}