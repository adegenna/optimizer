#include "Optimizer.hpp"
#include <iostream>
#include <fstream>
#include <unordered_map>


void NewtonRhapson::solve() {

    auto x = x0_;
    solver_data_->push_back( x , cost_.eval(x) );
    for ( int i=0; i<maxIters_; i++ ) {
        Eigen::VectorXd b   = cost_.H(x).transpose() * cost_.J(x);
        Eigen::MatrixXd HTH = cost_.H(x).transpose() * cost_.H(x);
        Eigen::VectorXd dx  = HTH.ldlt().solve( -b );
        for ( int j=0; j<x.size(); j++ ){
            x[j] += dx[j];
        }
        solver_data_->push_back( x , cost_.eval(x) );
    }

}


void GradientDescent::solve() {

    auto x = x0_;
    solver_data_->push_back( x , cost_.eval(x) );
    for ( int i=0; i<maxIters_; i++ ) {
        Eigen::VectorXd dx = -cost_.J(x) * scale_;
        for ( int j=0; j<x.size(); j++ ){
            x[j] += dx(j);
        }
        solver_data_->push_back( x , cost_.eval(x) );
    }
}


void GradientDescentMomentum::solve() {

    int n = 1;
    int d = cost_.get_dimn();
    auto x = x0_;
    solver_data_->push_back( x , cost_.eval(x) );
    Eigen::MatrixXd J_prev = Eigen::MatrixXd::Zero(n,d);
    Eigen::MatrixXd J_i    = cost_.J(x).transpose();
    for ( int i=0; i<maxIters_; i++ ) {
        Eigen::MatrixXd J_avg = weight_history_ * J_prev + J_i;
        Eigen::VectorXd dx = -J_avg.transpose() * scale_;
        for ( int j=0; j<x.size(); j++ ){
            x[j] += dx(j);
        }
        solver_data_->push_back( x , cost_.eval(x) );
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
        solver_data_->push_back( x , cost_x );
    }

}


void ParticleSwarm::set_initial_data_() {
    
    x_.resize( params_.n_swarm_ );
    vel_.resize( params_.n_swarm_ );
    p_.resize( params_.n_swarm_ );
    J_p_.resize( params_.n_swarm_ );
    for ( int i=0; i<params_.n_swarm_; i++ ) {
        x_[i]   = params_.draw_random_state_();
        vel_[i] = params_.draw_random_vel_();
        p_[i]   = x_[i];
        J_p_[i] = cost_.eval( p_[i] );
        solver_data_->push_back( x_[i] , J_p_[i] );
    }
    g_    = x_[0]; 
    J_g_  = J_p_[0];
    for ( int i=0; i<params_.n_swarm_; i++ ) {
        if ( J_p_[i] < J_g_ ) {
            g_   = x_[i];
            J_g_ = J_p_[i];
        }
    }

}


void ParticleSwarm::solve() {

    int iter = 1;
    while( ( params_.is_done_(x_) == false ) && ( iter < maxIters_ ) ) {
        solver_data_->resize_plus_one();
        for ( int i=0; i<params_.n_swarm_; i++ ) {
            for ( int j=0; j<x_[0].size(); j++ ) {
                auto r_p = (float)rand() / RAND_MAX;
                auto r_g = (float)rand() / RAND_MAX;
                vel_[i][j] = params_.w_ * vel_[i][j] + 
                             params_.phi_p_ * r_p * ( p_[i][j] - x_[i][j] ) + 
                             params_.phi_g_ * r_g * (    g_[j] - x_[i][j] ); 
                x_[i][j]  += vel_[i][j];
            }
            auto J_i = cost_.eval( x_[i] );
            if ( J_i < J_p_[i] ) {
                p_[i]   = x_[i];
                J_p_[i] = J_i;
                if ( J_p_[i] < J_g_ ) {
                    g_   = p_[i];
                    J_g_ = J_p_[i];
                }
            }
            solver_data_->push_back( x_[i] , J_p_[i] );
        }
        iter += 1;
    }

}