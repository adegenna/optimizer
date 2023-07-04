#pragma once

#include <functional>
#include <vector>
#include <Eigen/Dense>

class Function {

    public:

        Function( const std::function<float(const std::vector<float>& x )>& func_handle , int d_input ) : 
                    func_(func_handle) , d_(d_input) {}
        Function( const std::function<float(const std::vector<float>& x )>& func_handle ,
                    const std::function<Eigen::MatrixXd(const std::vector<float>& x )>& df_handle , 
                    int d_input ) : 
                    func_(func_handle) , dfunc_(df_handle) , d_(d_input) {}
        Function( const std::function<float(const std::vector<float>& x )>& func_handle ,
                    const std::function<Eigen::MatrixXd(const std::vector<float>& x )>& df_handle , 
                    const std::function<Eigen::MatrixXd(const std::vector<float>& x )>& df2_handle , 
                    int d_input ) : 
                    func_(func_handle) , dfunc_(df_handle) , H_func_(df2_handle) , d_(d_input) {}

        float eval( const std::vector<float>& x ) const { return func_(x); }
        Eigen::MatrixXd J( const std::vector<float>& x ) const { return dfunc_(x); }
        Eigen::MatrixXd H( const std::vector<float>& x ) const { return H_func_(x); }
        int get_dimn() const { return d_; }

    private:

        const std::function<float(const std::vector<float>& x)> func_;
        const std::function<Eigen::MatrixXd(const std::vector<float>& x)> dfunc_;
        const std::function<Eigen::MatrixXd(const std::vector<float>& x)> H_func_;
        int d_;      // dimn of function input

};