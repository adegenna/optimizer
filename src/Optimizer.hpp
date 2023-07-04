#pragma once

#include <string>
#include <memory>

#include "Function.hpp"
#include "Data.hpp"
#include "SimAnnParams.hpp"

class Optimizer {

    public:

        virtual void solve() = 0;
        void output_solver_history( const std::string outFileName ) { solver_data_.output_history(outFileName); }
        float get_last_cost() { return solver_data_.get_last_cost(); }
        std::vector<float> get_last_state() { return solver_data_.get_last_state(); }

    protected:

        const Function& cost_;
        Optimizer( const Function& cost , const std::vector<float>& x0 , int maxIters ) : cost_(cost) , x0_(x0) , maxIters_(maxIters) {};
        DataOptimizer solver_data_;
        std::vector<float> x0_;
        int maxIters_;

};


class NewtonRhapson : public Optimizer {

    public:
        
        NewtonRhapson( const Function& cost , const std::vector<float>& x0 , int maxIters ) : Optimizer( cost , x0 , maxIters ) {};
        void solve();

};

class GradientDescent : public Optimizer {

    public:

        GradientDescent( const Function& cost , const std::vector<float>& x0 , int maxIters , float scale ) : Optimizer( cost , x0 , maxIters ) , scale_(scale) {};
        void solve();

    private:

        float scale_;

};


class GradientDescentMomentum : public Optimizer {

    public:

        GradientDescentMomentum( const Function& cost , const std::vector<float>& x0 , int maxIters , float scale , const float weight_history ) : Optimizer( cost , x0 , maxIters ) , scale_(scale) , weight_history_(weight_history) {};
        void solve();

    private:

        float scale_;
        float weight_history_;

};


class SimulatedAnnealing : public Optimizer {

    public:

        SimulatedAnnealing( const Function& cost , const std::vector<float>& x0 , int maxIters , SimAnnParams s ) : Optimizer( cost , x0 , maxIters ) , sim_ann_params_(s) {};
        void solve();

    private:

        SimAnnParams sim_ann_params_;

};