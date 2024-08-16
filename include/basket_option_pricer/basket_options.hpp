#pragma once

#include <Eigen/Dense>
#include <string>
#include <tuple>

namespace BOP::BasketOptionPricer {

std::tuple<double, double, double> tau_solver(double m1, double m2, double m3,
                                              double kappa);

struct Result {
    double tau;
    double kappa;
    double cp;
    double rho;
    double skew;
    std::vector<double> theta;
    std::vector<double> delta;
    std::vector<double> vega;
    std::string dist;

    friend std::ostream& operator<<(std::ostream& os, const Result& r);
};

Result calculate(double T, double K, double r, Eigen::Ref<Eigen::VectorXd> a,
                 Eigen::Ref<Eigen::VectorXd> F0,
                 Eigen::Ref<Eigen::MatrixXd> rho,
                 Eigen::Ref<Eigen::VectorXd> sigma,
                 std::string distIn = "auto");

}  // namespace BOP::BasketOptionPricer