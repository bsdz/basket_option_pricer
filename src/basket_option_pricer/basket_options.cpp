
#include "basket_option_pricer/basket_options.hpp"

#include <ceres/ceres.h>
#include <cost_function.h>
#include <problem.h>
#include <types.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

#include "basket_option_pricer/basket_options_gen.hpp"
using ceres::Problem, ceres::CostFunction, ceres::Solver, ceres::Solve;
using std::string;

namespace BOP::BasketOptionPricer {

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ",
                             ", ", "", "", " << ", ";");

class QuadraticCostFunction : public ceres::SizedCostFunction<3, 3> {
   public:
    QuadraticCostFunction(double m1, double m2, double m3, double kappa)
        : m1_(m1), m2_(m2), m3_(m3), kappa_(kappa) {}
    virtual ~QuadraticCostFunction() {}
    virtual bool Evaluate(double const* const* x, double* residuals,
                          double** jacobians) const {
        const double x0 = x[0][0];
        const double x1 = x[0][1];
        const double x2 = x[0][2];

        Eigen::Vector3d X(x0, x1, x2);

        auto Y = fn_tau_solver(m1_, m2_, m3_, kappa_, X);

        residuals[0] = Y(0, 0);
        residuals[1] = Y(1, 0);
        residuals[2] = Y(2, 0);

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            auto Z = fn_tau_solver_prime(kappa_, X);
            jacobians[0][0] = Z(0, 0);
            jacobians[0][1] = Z(0, 1);
            jacobians[0][2] = Z(0, 2);
            jacobians[0][3 + 0] = Z(1, 0);
            jacobians[0][3 + 1] = Z(1, 1);
            jacobians[0][3 + 2] = Z(1, 2);
            jacobians[0][6 + 0] = Z(2, 0);
            jacobians[0][6 + 1] = Z(2, 1);
            jacobians[0][6 + 2] = Z(2, 2);
        }
        return true;
    }
    double m1_;
    double m2_;
    double m3_;
    double kappa_;
};

std::tuple<double, double, double> tau_solver(double m1, double m2, double m3,
                                              double kappa) {
    // returns {tau, mu_top, sigma_top}
    Problem problem;

    // initial values
    double x[3] = {0, 1, 2.7};

    CostFunction* cf0 = new QuadraticCostFunction(m1, m2, m3, kappa);
    problem.AddResidualBlock(cf0, nullptr, x);

    Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 100000;
    // options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    if (summary.termination_type == ceres::CONVERGENCE) {
        return {x[0], std::log(x[1]), std::sqrt(2. * std::log(x[2]))};
    } else {
        DLOG(INFO) << summary.BriefReport();
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }
}

std::ostream& operator<<(std::ostream& os, const Result& r) {
    os << "Results:\n"
       << "call price = " << r.cp << '\n'
       << "tau" << r.tau << '\n'
       << "kappa" << r.kappa << '\n'
       << "skew" << r.skew << '\n'
       << "dist" << r.dist << '\n';
    int N = r.theta.size();
    for (int l : std::views::iota(0, N)) {
        os << "theta" << l << " = " << r.theta[l] << '\n';
        os << "delta" << l << " = " << r.delta[l] << '\n';
        os << "vega" << l << " = " << r.vega[l] << '\n';
    }
    return os;
}

double fn_m3_T0(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
                int N) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            for (int k : std::views::iota(0, N))
                res += F0(i) * F0(j) * F0(k) * a(i) * a(j) * a(k);
    return res;
}

Result calculate(double T, double K, double r, Eigen::Ref<Eigen::VectorXd> a,
                 Eigen::Ref<Eigen::VectorXd> F0,
                 Eigen::Ref<Eigen::MatrixXd> rho,
                 Eigen::Ref<Eigen::VectorXd> sigma, string distIn) {
    Result res;

    auto N = F0.size();
    auto m3T0 = fn_m3_T0(F0, a, N);
    auto m1 = fn_m1(F0, a, N);
    auto m2 = fn_m2(F0, a, N, T, sigma, rho);
    auto m3 = fn_m3(F0, a, N, T, sigma, rho);

    double tau, mu_top, sigma_top;
    string dist;

    auto skew = (m3 - 3 * m2 + 3 * m1 - m3T0) / std::pow(m2 - m1 * m1, 3. / 2.);
    double kappa = skew < 0 ? -1. : 1.;
    std::tie(tau, mu_top, sigma_top) = tau_solver(m1, m2, m3, kappa);

    if (distIn == "auto") {
        std::vector<std::string> dist_parts;
        if (kappa < 0) {
            dist_parts.push_back("N");
        }
        if (tau < -1) {
            dist_parts.push_back("S");
        } else {
            tau = 0;
            mu_top = fn_mu(m1, m2, tau, kappa);
            sigma_top = fn_sigma(m1, m2, tau, kappa);
        }
        dist = std::accumulate(
                   dist_parts.begin(), dist_parts.end(), std::string(""),
                   [](std::string a, std::string b) { return a + b; }) +
               "LN";

    } else {
        dist = distIn;
    }

    double V, d1, d2, c;
    if (dist == "LN") {
        V = fn_V_ln(m1, m2);
        d1 = fn_d1_ln(m1, K, T, V);
        d2 = fn_d2_ln(m1, K, T, V);
        c = fn_P_ln(m1, K, T, r, d1, d2);
    } else if (dist == "NLN") {
        V = fn_V_nln(m1, m2);
        d1 = fn_d1_nln(m1, K, T, V);
        d2 = fn_d2_nln(m1, K, T, V);
        c = fn_P_nln(m1, K, T, r, d1, d2);
    } else if (dist == "SLN") {
        V = fn_V_sln(m1, m2, tau);
        d1 = fn_d1_sln(m1, K, T, tau, V);
        d2 = fn_d2_sln(m1, K, T, tau, V);
        c = fn_P_sln(m1, K, T, tau, r, d1, d2);
    } else if (dist == "NSLN") {
        V = fn_V_nsln(m1, m2, tau);
        d1 = fn_d1_nsln(m1, K, T, tau, V);
        d2 = fn_d2_nsln(m1, K, T, tau, V);
        c = fn_P_nsln(m1, K, T, tau, r, d1, d2);
    }

    DLOG(INFO) << "dist: " << dist;
    DLOG(INFO) << "m1: " << m1 << "; m2: " << m2 << "; m3: " << m3;
    DLOG(INFO) << "skew: " << skew << "; kappa: " << kappa << "; tau: " << tau;
    DLOG(INFO) << "mu: " << mu_top << "; sigma: " << sigma_top;
    DLOG(INFO) << "V: " << V << ";d1: " << d1 << "; d2: " << d2;
    DLOG(INFO) << "call price: " << c;

    res.kappa = kappa;
    res.tau = tau;
    res.cp = c;
    res.dist = dist;
    res.skew = skew;

    // greek calcs

    double rho_g;
    if (dist == "LN") {
        rho_g = fn_rho_ln(T, c);
    } else if (dist == "NLN") {
        rho_g = fn_rho_nln(T, c);
    } else if (dist == "SLN") {
        rho_g = fn_rho_sln(T, c);
    } else if (dist == "NSLN") {
        rho_g = fn_rho_nsln(T, c);
    }
    DLOG(INFO) << "rho: " << rho_g;
    res.rho = rho_g;

    auto J_top = fn_jacob_top(tau, kappa, mu_top, sigma_top);
    auto J_top_inv = J_top.inverse();
    DLOG(INFO) << "J_top: " << J_top.format(CommaInitFmt);

    for (int l : std::views::iota(0, N)) {
        auto jc = fn_jacob_comp(F0, a, N, T, sigma, rho, l);
        auto jp = J_top_inv * jc;

        DLOG(INFO) << "[" << l
                   << "]: " << "J_comp: " << jc.format(CommaInitFmt);
        DLOG(INFO) << "[" << l
                   << "]: " << "J_param: " << jp.format(CommaInitFmt);

        auto tau_F = jp(0, 0);
        auto tau_sigma = jp(0, 1);
        auto tau_T = jp(0, 2);
        auto m1_F = jc(0, 0);
        auto m2_F = jc(1, 0);
        auto m2_T = jc(1, 2);
        auto m2_sigma = jc(1, 1);

        double V_T, V_F, V_s, theta, delta, vega;
        if (dist == "LN") {
            V_T = fn_V_T_ln(m2, V, m2_T);
            V_F = fn_V_F_ln(m1, m2, V, m1_F, m2_F);
            V_s = fn_V_s_ln(m2, V, m2_sigma);
            theta = fn_theta_ln(K, T, r, d2, V_T, c);
            delta = fn_delta_ln(K, T, r, d1, d2, m1_F, V_F);
            vega = fn_vega_ln(K, T, r, d2, V_s);
        } else if (dist == "NLN") {
            V_T = fn_V_T_nln(m2, V, m2_T);
            V_F = fn_V_F_nln(m1, m2, V, m1_F, m2_F);
            V_s = fn_V_s_nln(m2, V, m2_sigma);
            theta = fn_theta_nln(K, T, r, d2, V_T, c);
            delta = fn_delta_nln(K, T, r, d1, d2, m1_F, V_F);
            vega = fn_vega_nln(K, T, r, d2, V_s);
        } else if (dist == "SLN") {
            V_T = fn_V_T_sln(m1, m2, tau, V, tau_T, m2_T);
            V_F = fn_V_F_sln(m1, m2, tau, V, tau_F, m1_F, m2_F);
            V_s = fn_V_s_sln(m1, m2, tau, V, tau_sigma, m2_sigma);
            theta = fn_theta_sln(K, T, tau, r, d1, d2, tau_T, V_T, c);
            delta = fn_delta_sln(K, T, tau, r, d1, d2, tau_F, m1_F, V_F);
            vega = fn_vega_sln(K, T, tau, r, d1, d2, tau_sigma, V_s);
        } else if (dist == "NSLN") {
            V_T = fn_V_T_nsln(m1, m2, tau, V, tau_T, m2_T);
            V_F = fn_V_F_nsln(m1, m2, tau, V, tau_F, m1_F, m2_F);
            V_s = fn_V_s_nsln(m1, m2, tau, V, tau_sigma, m2_sigma);
            theta = fn_theta_nsln(K, T, tau, r, d1, d2, tau_T, V_T, c);
            delta = fn_delta_nsln(K, T, tau, r, d1, d2, tau_F, m1_F, V_F);
            vega = fn_vega_nsln(K, T, tau, r, d1, d2, tau_sigma, V_s);
        }

        DLOG(INFO) << "[" << l << "]: " << "V_T: " << V_T << "; V_F: " << V_F
                   << "; V_s: " << V_s;
        DLOG(INFO) << "[" << l << "]: " << "theta: " << theta
                   << "; delta: " << delta << "; vega: " << vega;

        res.theta.push_back(theta);
        res.delta.push_back(delta);
        res.vega.push_back(vega);
    }

    return res;
}

}  // namespace BOP::BasketOptionPricer