#pragma once

#include <vector>
#include <string>

class DataOptimizer {

    public:

        virtual std::vector<float> get_last_state( ) = 0;
        virtual float get_last_cost( ) = 0;
        virtual void output_history( const std::string outFileName ) = 0;
        virtual void push_back( const std::vector<float>& x , float ) = 0;
        virtual void resize_plus_one( ) = 0;
    
};


class DataOptimizer_singleInitialValue : public DataOptimizer {

    public:

        std::vector<float> get_state( int idx ) { return history_state_[idx]; }
        float get_cost( int idx ) { return history_cost_[idx]; }
        std::vector<float> get_last_state( ) { return history_state_[history_state_.size()-1]; }
        float get_last_cost( ) { return history_cost_[history_cost_.size()-1]; }
        void push_back( const std::vector<float>& r , float J ) { 
            history_state_.push_back(r); history_cost_.push_back(J); 
        }
        void output_history( const std::string outFileName );
        void resize_plus_one( ) {
            history_state_.resize( history_state_.size()+1 );
            history_cost_.resize( history_cost_.size()+1 );
        }

    
    private:

        std::vector<std::vector<float>> history_state_;
        std::vector<float> history_cost_;

};

class DataOptimizer_ParticleMethod : public DataOptimizer {

    public:

        DataOptimizer_ParticleMethod() { history_state_.reserve(1); history_cost_.reserve(1); };
        std::vector<float> get_last_state( );
        float get_last_cost( );
        void output_history( const std::string outFileName );
        void push_back( const std::vector<float>& r , float J ) { 
            history_state_[history_state_.size()-1].push_back(r); 
            history_cost_[history_cost_.size()-1].push_back(J); 
        }
        void resize_plus_one( ) {
            history_state_.resize( history_state_.size()+1 );
            history_cost_.resize( history_cost_.size()+1 );
        }
    
    private:

        std::vector<std::vector<std::vector<float>>> history_state_;
        std::vector<std::vector<float>> history_cost_;

};
