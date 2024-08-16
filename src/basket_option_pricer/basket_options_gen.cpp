#include "basket_option_pricer/basket_options_gen.hpp"

#include <basket_option_pricer/standard_normal.hpp>
#include <cmath>
#include <ranges>

double sum_0(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N) {
    double res = 0;
    for (int i : std::views::iota(0, N)) res += F0(i) * a(i);
    return res;
}

double fn_m1(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N) {
    return sum_0(F0, a, N);
}

double sum_1(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            res += F0(i) * F0(j) * a(i) * a(j) *
                   std::exp(T * rho(i, j) * sigma(i) * sigma(j));
    return res;
}

double fn_m2(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho) {
    return sum_1(F0, a, N, T, sigma, rho);
}

double sum_2(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            for (int k : std::views::iota(0, N))
                res += F0(i) * F0(j) * F0(k) * a(i) * a(j) * a(k) *
                       std::exp(T * rho(i, j) * sigma(i) * sigma(j)) *
                       std::exp(T * rho(i, k) * sigma(i) * sigma(k)) *
                       std::exp(T * rho(j, k) * sigma(j) * sigma(k));
    return res;
}

double fn_m3(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho) {
    return sum_2(F0, a, N, T, sigma, rho);
}

double fn_V_ln(double m1, double m2) {
    return std::pow(std::log(m2 * (std::pow(m1, -2.0))), 0.5);
}

double fn_d1_ln(double F, double K, double T, double sigma) {
    return (std::pow(T, -1.0 / 2.0)) * (std::pow(sigma, -1.0)) *
           ((0.5) * T * (std::pow(sigma, 2.0)) +
            std::log(F * (std::pow(K, -1.0))));
}

double fn_d2_ln(double F, double K, double T, double sigma) {
    return (-1.0) * sigma * (std::pow(T, 0.5)) +
           (std::pow(T, -1.0 / 2.0)) * (std::pow(sigma, -1.0)) *
               ((0.5) * T * (std::pow(sigma, 2.0)) +
                std::log(F * (std::pow(K, -1.0))));
}

double fn_P_ln(double F, double K, double T, double r, double d1, double d2) {
    return (F * Phi(d1) + (-1.0) * K * Phi(d2)) * std::exp((-1.0) * T * r);
}

double fn_V_nln(double m1, double m2) {
    return std::pow(std::log(m2 * (std::pow(m1, -2.0))), 0.5);
}

double fn_d1_nln(double F, double K, double T, double sigma) {
    return (std::pow(T, -1.0 / 2.0)) * (std::pow(sigma, -1.0)) *
           ((0.5) * T * (std::pow(sigma, 2.0)) +
            std::log(F * (std::pow(K, -1.0))));
}

double fn_d2_nln(double F, double K, double T, double sigma) {
    return (-1.0) * sigma * (std::pow(T, 0.5)) +
           (std::pow(T, -1.0 / 2.0)) * (std::pow(sigma, -1.0)) *
               ((0.5) * T * (std::pow(sigma, 2.0)) +
                std::log(F * (std::pow(K, -1.0))));
}

double fn_P_nln(double F, double K, double T, double r, double d1, double d2) {
    return (-1.0) * (K * Phi((-1.0) * d2) + (-1.0) * F * Phi((-1.0) * d1)) *
           std::exp((-1.0) * T * r);
}

double fn_V_sln(double m1, double m2, double tau) {
    return std::pow(std::log((std::pow(tau + (-1.0) * m1, -2.0)) *
                             (m2 + std::pow(tau, 2.0) + (-2.0) * m1 * tau)),
                    0.5);
}

double fn_d1_sln(double F, double K, double T, double tau, double sigma) {
    return (std::pow(T, -1.0 / 2.0)) * (std::pow(sigma, -1.0)) *
           ((0.5) * T * (std::pow(sigma, 2.0)) +
            std::log((std::pow(K + (-1.0) * tau, -1.0)) * (F + (-1.0) * tau)));
}

double fn_d2_sln(double F, double K, double T, double tau, double sigma) {
    return (-1.0) * sigma * (std::pow(T, 0.5)) +
           (std::pow(T, -1.0 / 2.0)) * (std::pow(sigma, -1.0)) *
               ((0.5) * T * (std::pow(sigma, 2.0)) +
                std::log((std::pow(K + (-1.0) * tau, -1.0)) *
                         (F + (-1.0) * tau)));
}

double fn_P_sln(double F, double K, double T, double tau, double r, double d1,
                double d2) {
    return ((F + (-1.0) * tau) * Phi(d1) +
            (-1.0) * (K + (-1.0) * tau) * Phi(d2)) *
           std::exp((-1.0) * T * r);
}

double fn_V_nsln(double m1, double m2, double tau) {
    return std::pow(std::log((std::pow((-1.0) * m1 + (-1.0) * tau, -2.0)) *
                             (m2 + std::pow(tau, 2.0) + (2.0) * m1 * tau)),
                    0.5);
}

double fn_d1_nsln(double F, double K, double T, double tau, double sigma) {
    return (std::pow(T, -1.0 / 2.0)) * (std::pow(sigma, -1.0)) *
           ((0.5) * T * (std::pow(sigma, 2.0)) +
            std::log((std::pow((-1.0) * K + (-1.0) * tau, -1.0)) *
                     ((-1.0) * F + (-1.0) * tau)));
}

double fn_d2_nsln(double F, double K, double T, double tau, double sigma) {
    return (-1.0) * sigma * (std::pow(T, 0.5)) +
           (std::pow(T, -1.0 / 2.0)) * (std::pow(sigma, -1.0)) *
               ((0.5) * T * (std::pow(sigma, 2.0)) +
                std::log((std::pow((-1.0) * K + (-1.0) * tau, -1.0)) *
                         ((-1.0) * F + (-1.0) * tau)));
}

double fn_P_nsln(double F, double K, double T, double tau, double r, double d1,
                 double d2) {
    return (-1.0) *
           (((-1.0) * F + (-1.0) * tau) * Phi((-1.0) * d1) +
            (-1.0) * ((-1.0) * K + (-1.0) * tau) * Phi((-1.0) * d2)) *
           std::exp((-1.0) * T * r);
}

double fn_mu(double m1, double m2, double tau, double kappa) {
    return std::log(
        std::abs((-1.0) * (std::pow(kappa, -1.0)) *
                 (std::pow(m1 + (-1.0) * kappa * tau, 2.0)) *
                 (std::pow(m2 + (std::pow(kappa, 2.0)) * (std::pow(tau, 2.0)) +
                               (-2.0) * kappa * m1 * tau,
                           -1.0 / 2.0))));
}

double fn_sigma(double m1, double m2, double tau, double kappa) {
    return std::pow(
        std::log((std::pow((-1.0) * m1 + kappa * tau, -2.0)) *
                 (m2 + (std::pow(kappa, 2.0)) * (std::pow(tau, 2.0)) +
                  (-2.0) * kappa * m1 * tau)),
        0.5);
}

double sum_3(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho, int l) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        res += F0(i) * a(i) * std::exp(T * rho(l, i) * sigma(i) * sigma(l));
    return res;
}

double sum_4(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho, int l) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        res += F0(i) * a(i) * rho(l, i) * sigma(i) *
               std::exp(T * rho(l, i) * sigma(i) * sigma(l));
    return res;
}

double sum_5(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            res += F0(i) * F0(j) * a(i) * a(j) * rho(i, j) * sigma(i) *
                   sigma(j) * std::exp(T * rho(i, j) * sigma(i) * sigma(j));
    return res;
}

double sum_6(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho, int l) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            res += F0(i) * F0(j) * a(i) * a(j) *
                   std::exp(T * rho(i, j) * sigma(i) * sigma(j)) *
                   std::exp(T * rho(l, i) * sigma(i) * sigma(l)) *
                   std::exp(T * rho(l, j) * sigma(j) * sigma(l));
    return res;
}

double sum_7(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho, int l) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            res += F0(i) * F0(j) * a(i) * a(j) * rho(l, i) * sigma(i) *
                   std::exp(T * rho(i, j) * sigma(i) * sigma(j)) *
                   std::exp(T * rho(l, i) * sigma(i) * sigma(l)) *
                   std::exp(T * rho(l, j) * sigma(j) * sigma(l));
    return res;
}

double sum_8(Eigen::Ref<Eigen::VectorXd> F0, Eigen::Ref<Eigen::VectorXd> a,
             int N, double T, Eigen::Ref<Eigen::VectorXd> sigma,
             Eigen::Ref<Eigen::MatrixXd> rho) {
    double res = 0;
    for (int i : std::views::iota(0, N))
        for (int j : std::views::iota(0, N))
            for (int k : std::views::iota(0, N))
                res += (rho(i, j) * sigma(i) * sigma(j) +
                        rho(i, k) * sigma(i) * sigma(k) +
                        rho(j, k) * sigma(j) * sigma(k)) *
                       F0(i) * F0(j) * F0(k) * a(i) * a(j) * a(k) *
                       std::exp(T * rho(i, j) * sigma(i) * sigma(j)) *
                       std::exp(T * rho(i, k) * sigma(i) * sigma(k)) *
                       std::exp(T * rho(j, k) * sigma(j) * sigma(k));
    return res;
}

Eigen::MatrixXd idm_0(Eigen::Ref<Eigen::VectorXd> F0,
                      Eigen::Ref<Eigen::VectorXd> a, int N, double T,
                      Eigen::Ref<Eigen::VectorXd> sigma,
                      Eigen::Ref<Eigen::MatrixXd> rho, int l) {
    Eigen::MatrixXd mat(3, 3);
    mat << a(l), 0.0, 0.0, (2.0) * a(l) * (sum_3(F0, a, N, T, sigma, rho, l)),
        (2.0) * T * F0(l) * a(l) * (sum_4(F0, a, N, T, sigma, rho, l)),
        sum_5(F0, a, N, T, sigma, rho),
        (3.0) * a(l) * (sum_6(F0, a, N, T, sigma, rho, l)),
        (6.0) * T * F0(l) * a(l) * (sum_7(F0, a, N, T, sigma, rho, l)),
        sum_8(F0, a, N, T, sigma, rho);
    return mat;
}

Eigen::MatrixXd fn_jacob_comp(Eigen::Ref<Eigen::VectorXd> F0,
                              Eigen::Ref<Eigen::VectorXd> a, int N, double T,
                              Eigen::Ref<Eigen::VectorXd> sigma,
                              Eigen::Ref<Eigen::MatrixXd> rho, int l) {
    return idm_0(F0, a, N, T, sigma, rho, l);
}

Eigen::MatrixXd idm_1(double tau, double kappa, double mu, double sigma) {
    Eigen::MatrixXd mat(3, 3);
    mat << kappa, kappa * std::exp(mu + (0.5) * (std::pow(sigma, 2.0))),
        kappa * sigma * std::exp(mu + (0.5) * (std::pow(sigma, 2.0))),
        (std::pow(kappa, 2.0)) *
            ((2.0) * tau +
             (2.0) * std::exp(mu + (0.5) * (std::pow(sigma, 2.0)))),
        (std::pow(kappa, 2.0)) *
            ((2.0) * std::exp((2.0) * mu + (2.0) * (std::pow(sigma, 2.0))) +
             (2.0) * tau * std::exp(mu + (0.5) * (std::pow(sigma, 2.0)))),
        (std::pow(kappa, 2.0)) *
            ((4.0) * sigma *
                 std::exp((2.0) * mu + (2.0) * (std::pow(sigma, 2.0))) +
             (2.0) * sigma * tau *
                 std::exp(mu + (0.5) * (std::pow(sigma, 2.0)))),
        (std::pow(kappa, 3.0)) *
            ((3.0) * (std::pow(tau, 2.0)) +
             (3.0) * std::exp((2.0) * mu + (2.0) * (std::pow(sigma, 2.0))) +
             (6.0) * tau * std::exp(mu + (0.5) * (std::pow(sigma, 2.0)))),
        (std::pow(kappa, 3.0)) *
            ((3.0) *
                 std::exp((3.0) * mu + (9.0 / 2.0) * (std::pow(sigma, 2.0))) +
             (3.0) * (std::pow(tau, 2.0)) *
                 std::exp(mu + (0.5) * (std::pow(sigma, 2.0))) +
             (6.0) * tau *
                 std::exp((2.0) * mu + (2.0) * (std::pow(sigma, 2.0)))),
        (std::pow(kappa, 3.0)) *
            ((9.0) * sigma *
                 std::exp((3.0) * mu + (9.0 / 2.0) * (std::pow(sigma, 2.0))) +
             (3.0) * sigma * (std::pow(tau, 2.0)) *
                 std::exp(mu + (0.5) * (std::pow(sigma, 2.0))) +
             (12.0) * sigma * tau *
                 std::exp((2.0) * mu + (2.0) * (std::pow(sigma, 2.0))));
    return mat;
}

Eigen::MatrixXd fn_jacob_top(double tau, double kappa, double mu,
                             double sigma) {
    return idm_1(tau, kappa, mu, sigma);
}

double fn_rho_ln(double T, double c) { return (-1.0) * c * T; }

double fn_theta_ln(double K, double T, double r, double d2, double V_T,
                   double c) {
    return (-1.0) * c * r + V_T * K * std::exp((-1.0) * T * r) * phi(d2);
}

double fn_delta_ln(double K, double T, double r, double d1, double d2,
                   double M1_F, double V_F) {
    return (M1_F * Phi(d1) + V_F * K * phi(d2)) * std::exp((-1.0) * T * r);
}

double fn_vega_ln(double K, double T, double r, double d2, double V_sigma) {
    return V_sigma * K * std::exp((-1.0) * T * r) * phi(d2);
}

double fn_V_T_ln(double M2, double V, double M2_T) {
    return (0.5) * M2_T * (std::pow(M2, -1.0)) * (std::pow(V, -1.0));
}

double fn_V_F_ln(double M1, double M2, double V, double M1_F, double M2_F) {
    return (-1.0 / 2.0) * (std::pow(M1, -1.0)) * (std::pow(M2, -1.0)) *
           (std::pow(V, -1.0)) * ((-1.0) * M1 * M2_F + (2.0) * M1_F * M2);
}

double fn_V_s_ln(double M2, double V, double M2_sigma) {
    return (0.5) * M2_sigma * (std::pow(M2, -1.0)) * (std::pow(V, -1.0));
}

double fn_rho_nln(double T, double c) { return (-1.0) * c * T; }

double fn_theta_nln(double K, double T, double r, double d2, double V_T,
                    double c) {
    return (-1.0) * c * r +
           (-1.0) * V_T * K * std::exp((-1.0) * T * r) * phi((-1.0) * d2);
}

double fn_delta_nln(double K, double T, double r, double d1, double d2,
                    double M1_F, double V_F) {
    return (M1_F * Phi((-1.0) * d1) + (-1.0) * V_F * K * phi((-1.0) * d2)) *
           std::exp((-1.0) * T * r);
}

double fn_vega_nln(double K, double T, double r, double d2, double V_sigma) {
    return (-1.0) * V_sigma * K * std::exp((-1.0) * T * r) * phi((-1.0) * d2);
}

double fn_V_T_nln(double M2, double V, double M2_T) {
    return (0.5) * M2_T * (std::pow(M2, -1.0)) * (std::pow(V, -1.0));
}

double fn_V_F_nln(double M1, double M2, double V, double M1_F, double M2_F) {
    return (-1.0 / 2.0) * (std::pow(M1, -1.0)) * (std::pow(M2, -1.0)) *
           (std::pow(V, -1.0)) * ((-1.0) * M1 * M2_F + (2.0) * M1_F * M2);
}

double fn_V_s_nln(double M2, double V, double M2_sigma) {
    return (0.5) * M2_sigma * (std::pow(M2, -1.0)) * (std::pow(V, -1.0));
}

double fn_rho_sln(double T, double c) { return (-1.0) * c * T; }

double fn_theta_sln(double K, double T, double tau, double r, double d1,
                    double d2, double tau_T, double V_T, double c) {
    return ((-1.0) * tau_T * ((-1.0) * Phi(d2) + Phi(d1)) +
            V_T * (K + (-1.0) * tau) * phi(d2) +
            (-1.0) * c * r * std::exp(T * r)) *
           std::exp((-1.0) * T * r);
}

double fn_delta_sln(double K, double T, double tau, double r, double d1,
                    double d2, double tau_F, double M1_F, double V_F) {
    return (tau_F * Phi(d2) + (M1_F + (-1.0) * tau_F) * Phi(d1) +
            V_F * (K + (-1.0) * tau) * phi(d2)) *
           std::exp((-1.0) * T * r);
}

double fn_vega_sln(double K, double T, double tau, double r, double d1,
                   double d2, double tau_sigma, double V_sigma) {
    return (tau_sigma * Phi(d2) + (-1.0) * tau_sigma * Phi(d1) +
            V_sigma * (K + (-1.0) * tau) * phi(d2)) *
           std::exp((-1.0) * T * r);
}

double fn_V_T_sln(double M1, double M2, double tau, double V, double tau_T,
                  double M2_T) {
    return (-1.0 / 2.0) * (std::pow(V, -1.0)) *
           (std::pow(M1 + (-1.0) * tau, -1.0)) *
           (std::pow(
               (-1.0) * M2 + (-1.0) * (std::pow(tau, 2.0)) + (2.0) * M1 * tau,
               -1.0)) *
           (M1 * M2_T + (-1.0) * M2_T * tau +
            (-2.0) * tau_T * (std::pow(M1, 2.0)) + (2.0) * M2 * tau_T);
}

double fn_V_F_sln(double M1, double M2, double tau, double V, double tau_F,
                  double M1_F, double M2_F) {
    return (-1.0 / 2.0) * (std::pow(V, -1.0)) *
           (std::pow(M1 + (-1.0) * tau, -1.0)) *
           (std::pow(
               (-1.0) * M2 + (-1.0) * (std::pow(tau, 2.0)) + (2.0) * M1 * tau,
               -1.0)) *
           (M1 * M2_F + (-1.0) * M2_F * tau + (-2.0) * M1_F * M2 +
            (-2.0) * tau_F * (std::pow(M1, 2.0)) + (2.0) * M2 * tau_F +
            (2.0) * M1 * M1_F * tau);
}

double fn_V_s_sln(double M1, double M2, double tau, double V, double tau_sigma,
                  double M2_sigma) {
    return (-1.0 / 2.0) * (std::pow(V, -1.0)) *
           (std::pow(M1 + (-1.0) * tau, -1.0)) *
           (std::pow(
               (-1.0) * M2 + (-1.0) * (std::pow(tau, 2.0)) + (2.0) * M1 * tau,
               -1.0)) *
           (M1 * M2_sigma + (-1.0) * M2_sigma * tau +
            (-2.0) * tau_sigma * (std::pow(M1, 2.0)) + (2.0) * M2 * tau_sigma);
}

double fn_rho_nsln(double T, double c) { return (-1.0) * c * T; }

double fn_theta_nsln(double K, double T, double tau, double r, double d1,
                     double d2, double tau_T, double V_T, double c) {
    return (tau_T * ((-1.0) * Phi((-1.0) * d2) + Phi((-1.0) * d1)) +
            (-1.0) * V_T * (tau + K) * phi((-1.0) * d2) +
            (-1.0) * c * r * std::exp(T * r)) *
           std::exp((-1.0) * T * r);
}

double fn_delta_nsln(double K, double T, double tau, double r, double d1,
                     double d2, double tau_F, double M1_F, double V_F) {
    return ((M1_F + tau_F) * Phi((-1.0) * d1) +
            (-1.0) * tau_F * Phi((-1.0) * d2) +
            (-1.0) * V_F * (tau + K) * phi((-1.0) * d2)) *
           std::exp((-1.0) * T * r);
}

double fn_vega_nsln(double K, double T, double tau, double r, double d1,
                    double d2, double tau_sigma, double V_sigma) {
    return (tau_sigma * Phi((-1.0) * d1) +
            (-1.0) * tau_sigma * Phi((-1.0) * d2) +
            (-1.0) * V_sigma * (tau + K) * phi((-1.0) * d2)) *
           std::exp((-1.0) * T * r);
}

double fn_V_T_nsln(double M1, double M2, double tau, double V, double tau_T,
                   double M2_T) {
    return (0.5) * (std::pow(V, -1.0)) * (std::pow(M1 + tau, -1.0)) *
           (std::pow(M2 + std::pow(tau, 2.0) + (2.0) * M1 * tau, -1.0)) *
           (M1 * M2_T + M2_T * tau + (-2.0) * M2 * tau_T +
            (2.0) * tau_T * (std::pow(M1, 2.0)));
}

double fn_V_F_nsln(double M1, double M2, double tau, double V, double tau_F,
                   double M1_F, double M2_F) {
    return (-1.0 / 2.0) * (std::pow(V, -1.0)) * (std::pow(M1 + tau, -1.0)) *
           (std::pow(M2 + std::pow(tau, 2.0) + (2.0) * M1 * tau, -1.0)) *
           ((-1.0) * M1 * M2_F + (-1.0) * M2_F * tau +
            (-2.0) * tau_F * (std::pow(M1, 2.0)) + (2.0) * M1_F * M2 +
            (2.0) * M2 * tau_F + (2.0) * M1 * M1_F * tau);
}

double fn_V_s_nsln(double M1, double M2, double tau, double V, double tau_sigma,
                   double M2_sigma) {
    return (0.5) * (std::pow(V, -1.0)) * (std::pow(M1 + tau, -1.0)) *
           (std::pow(M2 + std::pow(tau, 2.0) + (2.0) * M1 * tau, -1.0)) *
           (M1 * M2_sigma + M2_sigma * tau + (-2.0) * M2 * tau_sigma +
            (2.0) * tau_sigma * (std::pow(M1, 2.0)));
}

Eigen::MatrixXd idm_2(double m1, double m2, double m3, double kappa,
                      Eigen::Ref<Eigen::VectorXd> x) {
    Eigen::MatrixXd mat(3, 1);
    mat << m1 + (-1.0) * kappa * (x(1) * x(2) + x(0)),
        m2 + (-1.0) * (std::pow(kappa, 2.0)) *
                 (std::pow(x(0), 2.0) +
                  (std::pow(x(1), 2.0)) * (std::pow(x(2), 4.0)) +
                  (2.0) * x(0) * x(1) * x(2)),
        m3 + (-1.0) * (std::pow(kappa, 3.0)) *
                 (std::pow(x(0), 3.0) +
                  (std::pow(x(1), 3.0)) * (std::pow(x(2), 9.0)) +
                  (3.0) * (std::pow(x(0), 2.0)) * x(1) * x(2) +
                  (3.0) * (std::pow(x(1), 2.0)) * (std::pow(x(2), 4.0)) * x(0));
    return mat;
}

Eigen::MatrixXd fn_tau_solver(double m1, double m2, double m3, double kappa,
                              Eigen::Ref<Eigen::VectorXd> x) {
    return idm_2(m1, m2, m3, kappa, x);
}

Eigen::MatrixXd idm_3(double kappa, Eigen::Ref<Eigen::VectorXd> x) {
    Eigen::MatrixXd mat(3, 3);
    mat << (-1.0) * kappa, (-1.0) * kappa * x(2), (-1.0) * kappa * x(1),
        (-1.0) * (std::pow(kappa, 2.0)) * ((2.0) * x(0) + (2.0) * x(1) * x(2)),
        (-1.0) * (std::pow(kappa, 2.0)) *
            ((2.0) * (std::pow(x(2), 4.0)) * x(1) + (2.0) * x(0) * x(2)),
        (-1.0) * (std::pow(kappa, 2.0)) *
            ((2.0) * x(0) * x(1) +
             (4.0) * (std::pow(x(1), 2.0)) * (std::pow(x(2), 3.0))),
        (-1.0) * (std::pow(kappa, 3.0)) *
            ((3.0) * (std::pow(x(0), 2.0)) +
             (3.0) * (std::pow(x(1), 2.0)) * (std::pow(x(2), 4.0)) +
             (6.0) * x(0) * x(1) * x(2)),
        (-1.0) * (std::pow(kappa, 3.0)) *
            ((3.0) * (std::pow(x(0), 2.0)) * x(2) +
             (3.0) * (std::pow(x(1), 2.0)) * (std::pow(x(2), 9.0)) +
             (6.0) * (std::pow(x(2), 4.0)) * x(0) * x(1)),
        (-1.0) * (std::pow(kappa, 3.0)) *
            ((3.0) * (std::pow(x(0), 2.0)) * x(1) +
             (9.0) * (std::pow(x(1), 3.0)) * (std::pow(x(2), 8.0)) +
             (12.0) * (std::pow(x(1), 2.0)) * (std::pow(x(2), 3.0)) * x(0));
    return mat;
}

Eigen::MatrixXd fn_tau_solver_prime(double kappa,
                                    Eigen::Ref<Eigen::VectorXd> x) {
    return idm_3(kappa, x);
}
