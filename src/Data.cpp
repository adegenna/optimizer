#include "Data.hpp"
#include <iostream>
#include <fstream>

void DataOptimizer::output_history( const std::string outFileName ) {

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