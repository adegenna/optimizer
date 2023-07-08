#include "Data.hpp"
#include <iostream>
#include <fstream>

void DataOptimizer_singleInitialValue::output_history( const std::string outFileName ) {

    std::ofstream outfile( outFileName );
    outfile << "( x , cost(x) ) : " << std::endl;
    for ( int i=0; i<history_cost_.size(); i++ ){
        for ( int j=0; j<history_state_[i].size(); j++ ){
            outfile << history_state_[i][j] << ",";
        }
        outfile << " ; " << history_cost_[i] << std::endl;
    }
    outfile.close();
}


void DataOptimizer_ParticleMethod::output_history( const std::string outFileName ) {

    std::ofstream outfile( outFileName );
    outfile << "( x , cost(x) ) : " << std::endl;
    outfile << std::endl << "epoch : " << std::endl;
    for ( int i=0; i<history_cost_.size(); i++ ){ // n_epochs
        for ( int j=0; j<history_state_[i].size(); j++ ){ // n_particles
            for ( int k=0; k<history_state_[i][j].size(); k++ ) {
                outfile << history_state_[i][j][k] << ",";
            }
            outfile << " ; " << history_cost_[i][j] << std::endl;
        }
        outfile << std::endl << "epoch : " << std::endl;
    }
    outfile.close();
}

std::vector<float> DataOptimizer_ParticleMethod::get_last_state( ) {

    std::vector<float> min_state = history_state_[history_state_.size()-1][0]; // final epoch, first particle
    auto cost_min = history_cost_[history_cost_.size()-1][0];
    for ( int i=0; i<history_state_[history_state_.size()-1].size(); i++ ) { // particles in final epoch
        if ( history_cost_[history_state_.size()-1][i] < cost_min ) {
            min_state = history_state_[history_state_.size()-1][i];
            cost_min  = history_cost_[history_cost_.size()-1][i];
        }
    }
    return min_state;
}

float DataOptimizer_ParticleMethod::get_last_cost( ) {
    
    auto cost_min = history_cost_[history_cost_.size()-1][0];
    for ( int i=0; i<history_cost_[history_cost_.size()-1].size(); i++ ) { // particles in final epoch
        if ( history_cost_[history_state_.size()-1][i] < cost_min ) {
            cost_min  = history_cost_[history_cost_.size()-1][i];
        }
    }
    return cost_min;

}