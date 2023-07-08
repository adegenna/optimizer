#pragma once

#include <string>
#include <memory>

#include "Function.hpp"
#include "Data.hpp"
#include "SimAnnParams.hpp"
#include "PartSwarmParams.hpp"

class Optimizer {

    public:

        virtual void solve() = 0;
        void output_solver_history( const std::string outFileName ) { solver_data_->output_history(outFileName); }
        float get_last_cost() { return solver_data_->get_last_cost(); }
        std::vector<float> get_last_state() { return solver_data_->get_last_state(); }

    protected:

        const Function& cost_;
        Optimizer( const Function& cost , int maxIters , std::unique_ptr<DataOptimizer> dptr ) : cost_(cost) , maxIters_(maxIters) , solver_data_(std::move(dptr)) {};
        int maxIters_;
        std::unique_ptr<DataOptimizer> solver_data_;

};


class NewtonRhapson : public Optimizer {

    public:
        
        NewtonRhapson( const Function& cost , const std::vector<float>& x0 , int maxIters ) : Optimizer( cost , maxIters , std::make_unique<DataOptimizer_singleInitialValue>() ) , x0_(x0) {};
        void solve();

    private:
        
        std::vector<float> x0_;

};

class GradientDescent : public Optimizer {

    public:

        GradientDescent( const Function& cost , const std::vector<float>& x0 , int maxIters , float scale ) : Optimizer( cost , maxIters , std::make_unique<DataOptimizer_singleInitialValue>() ) , scale_(scale) , x0_(x0) {};
        void solve();

    private:

        float scale_;        
        std::vector<float> x0_;

};


class GradientDescentMomentum : public Optimizer {

    public:

        GradientDescentMomentum( const Function& cost , const std::vector<float>& x0 , int maxIters , float scale , const float weight_history ) : Optimizer( cost , maxIters , std::make_unique<DataOptimizer_singleInitialValue>() ) , scale_(scale) , weight_history_(weight_history) , x0_(x0) {};
        void solve();

    private:

        float scale_;
        float weight_history_;
        std::vector<float> x0_;

};


class SimulatedAnnealing : public Optimizer {

    public:

        SimulatedAnnealing( const Function& cost , const std::vector<float>& x0 , int maxIters , SimAnnParams s ) : Optimizer( cost , maxIters , std::make_unique<DataOptimizer_singleInitialValue>() ) , sim_ann_params_(s) , x0_(x0) {};
        void solve();

    private:

        SimAnnParams sim_ann_params_;
        std::vector<float> x0_;

};


class ParticleSwarm : public Optimizer {

    public:

        ParticleSwarm( const Function& cost , int maxIters , PartSwarmParams s ) : Optimizer( cost , maxIters , std::make_unique<DataOptimizer_ParticleMethod>() ) , params_(s) { set_initial_data_(); };
        void solve();

    private:

        PartSwarmParams params_;
        std::vector<std::vector<float>> x_;
        std::vector<std::vector<float>> vel_;
        std::vector<std::vector<float>> p_;
        std::vector<float> g_;
        std::vector<float> J_p_;
        float J_g_;
        void set_initial_data_();

};