#pragma once

#include <vector>
#include <string>

class DataOptimizer {

    public:

        std::vector<float> get_state( int idx ) { return history_state_[idx]; }
        float get_cost( int idx ) { return history_cost_[idx]; }
        std::vector<float> get_last_state( ) { return history_state_[history_state_.size()-1]; }
        float get_last_cost( ) { return history_cost_[history_cost_.size()-1]; }
        void push_back( const std::vector<float>& r , float J ) { 
            history_state_.push_back(r); history_cost_.push_back(J); 
        }
        void output_history( const std::string outFileName );
    
    private:

        std::vector<std::vector<float>> history_state_;
        std::vector<float> history_cost_;

};