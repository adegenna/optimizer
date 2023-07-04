#include "Optimizer.hpp"
#include <iostream>
#include <fstream>
#include <unordered_map>


void NewtonRhapson::solve() {

    auto x = x0_;
    solver_data_.push_back( x , cost_.eval(x) );
    for ( int i=0; i<maxIters_; i++ ) {
        Eigen::VectorXd b   = cost_.H(x).transpose() * cost_.J(x);
        Eigen::MatrixXd HTH = cost_.H(x).transpose() * cost_.H(x);
        Eigen::VectorXd dx  = HTH.ldlt().solve( -b );
        for ( int j=0; j<x.size(); j++ ){
            x[j] += dx[j];
        }
        solver_data_.push_back( x , cost_.eval(x) );
    }

}


void GradientDescent::solve() {

    auto x = x0_;
    solver_data_.push_back( x , cost_.eval(x) );
    for ( int i=0; i<maxIters_; i++ ) {
        Eigen::VectorXd dx = -cost_.J(x) * scale_;
        for ( int j=0; j<x.size(); j++ ){
            x[j] += dx(j);
        }
        solver_data_.push_back( x , cost_.eval(x) );
    }
}


void GradientDescentMomentum::solve() {

    int n = 1;
    int d = cost_.get_dimn();
    auto x = x0_;
    solver_data_.push_back( x , cost_.eval(x) );
    Eigen::MatrixXd J_prev = Eigen::MatrixXd::Zero(n,d);
    Eigen::MatrixXd J_i    = cost_.J(x).transpose();
    for ( int i=0; i<maxIters_; i++ ) {
        Eigen::MatrixXd J_avg = weight_history_ * J_prev + J_i;
        Eigen::VectorXd dx = -J_avg.transpose() * scale_;
        for ( int j=0; j<x.size(); j++ ){
            x[j] += dx(j);
        }
        solver_data_.push_back( x , cost_.eval(x) );
        J_prev = J_avg;
        J_i    = cost_.J(x).transpose();
    }
}


void SimulatedAnnealing::solve() {

    auto x = x0_;
    for ( int i=0; i<maxIters_; i++ ) {
        double T   = sim_ann_params_.f_annealing_( (float)i );
        auto x_new = sim_ann_params_.f_random_neighbor_( x );
        auto cost_x   = cost_.eval(x);
        auto cost_new = cost_.eval(x_new);
        if ( cost_new < cost_x ) {
            x      = x_new;
            cost_x = cost_new;
        }
        else if ( sim_ann_params_.prob_accept_( cost_x , cost_new , T ) >= (float)rand() ) {
            x      = x_new;
            cost_x = cost_new;
        }
        solver_data_.push_back( x , cost_x );
    }

}