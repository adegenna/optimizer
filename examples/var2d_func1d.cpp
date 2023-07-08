#include <math.h>
#include <Eigen/Dense>
#include "../src/Optimizer.hpp"
#include "../src/FactoryOptimizer.hpp"

float J( const std::vector<float>& x ) {
    return 0.5 * pow( x[0]-1 , 2 ) + pow( x[1] - 2 , 2 );
}

Eigen::MatrixXd DJ( const std::vector<float>& x ){
    Eigen::MatrixXd M(2,1); 
    M(0,0) = x[0] - 1;
    M(1,0) = 2*x[1] - 4;
    return M;
}


int main() {

    auto lambda_cost  = [](const std::vector<float>& vec) -> float { return J(vec); };
    auto lambda_Dcost = [](const std::vector<float>& vec) -> Eigen::MatrixXd { return DJ(vec); };
    std::vector<float> x0 = { 0.5 , 1 };
    auto scale = 0.1;
    Function cost( lambda_cost , lambda_Dcost , 2 );

    GradDesMomParams p;
    p.x0_       = x0;
    p.maxIters_ = 16;
    p.scale_    = scale;
    p.momentum_ = 0.7;

    OptimizerParams params = { .type = OptimizerParams::Type::GradientDescentMomentum,
            .cost_function = cost,
            .gdm_params    = p };

    auto rf = FactoryOptimizer::makeOptimizer(params);
    
    rf->solve();
    
    rf->output_solver_history( "rf_history.out" );

}