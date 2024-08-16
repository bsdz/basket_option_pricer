#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <string>

#include "basket_option_pricer/basket_options.hpp"
#include "basket_option_pricer/basket_options_gen.hpp"

using std::string, std::string_literals::operator""s, Eigen::Matrix2d,
    Eigen::VectorXd, Eigen::MatrixXd, Eigen::Matrix, Eigen::Vector3d,
    BOP::BasketOptionPricer::calculate, BOP::BasketOptionPricer::tau_solver;

template <class Base>
class EigenPrintWrap : public Base {
    friend std::ostream &operator<<(std::ostream &os, const EigenPrintWrap &m) {
        os << std::endl << static_cast<Base>(m) << std::endl;
        return os;
    }
};

template <class Base>
const EigenPrintWrap<Base> &print_wrap(const Base &base) {
    return static_cast<const EigenPrintWrap<Base> &>(base);
}

// helper for comparing Eigen types with ASSERT_PRED2
template <typename T>
inline bool is_approx(const T &lhs, const T &rhs) {
    return lhs.isApprox(rhs, 1e-8);
}

// for more convenient use
#define ASSERT_MATRIX_ALMOST_EQUAL(m1, m2) \
    ASSERT_PRED2(is_approx<Eigen::MatrixXd>, print_wrap(m1), print_wrap(m2))

TEST(Eigen, MatrixMultiplication) {
    Matrix2d A, B, C_expect, C_actual;

    A << 1., 2., 3., 4.;
    B << 5., 6., 7., 8.;
    C_expect << 19., 22., 43., 50.;

    C_actual = A * B;

    // ASSERT_TRUE(C_actual.isApprox(C_expect));
    ASSERT_MATRIX_ALMOST_EQUAL(C_actual, C_expect);
}

TEST(BOP, Borovkova1) {
    string dist = "SLN";
    double tp = 7.751;
    VectorXd F0{{100, 120}};
    VectorXd sigma{{0.2, 0.3}};
    VectorXd a{{-1, 1}};
    Matrix<double, 2, 2> rho{{1.0, 0.9}, {0.9, 1.0}};
    double K = 20;
    double T = 1;
    double r = 0.03;
    int N = F0.rows();

    EXPECT_NEAR(fn_m1(F0, a, N), 20, 1e-6);
    EXPECT_NEAR(fn_m2(F0, a, N, T, sigma, rho), 832.6, 0.02);
    EXPECT_NEAR(fn_m3(F0, a, N, T, sigma, rho), 44450.60488488572, 0.02);

    auto [tau, mu_top, sigma_top] =
        tau_solver(20, 832.5869755569875, 44450.60488488572, 1);

    Vector3d x0{{-35.95324953671213, std::exp(3.959804392149017),
                 std::exp(std::pow(0.3597558066805427, 2.) / 2.)}};
    auto foo = fn_tau_solver(20, 832.5869755569875, 44450.60488488572, 1, x0);

    auto res = calculate(T, K, r, a, F0, rho, sigma);
    std::cout << res << std::endl;
}

TEST(BOP, Borovkova2) {
    string dist = "NSLN";
    double tp = 16.910;
    VectorXd F0{{150, 100}};
    VectorXd sigma{{0.3, 0.2}};
    VectorXd a{{-1, 1}};
    Matrix<double, 2, 2> rho{{1.0, 0.3}, {0.3, 1.0}};
    double K = -50;
    double T = 1;
    double r = 0.03;

    auto res = calculate(T, K, r, a, F0, rho, sigma);
    // std::cout << res << std::endl;
}

TEST(BOP, Borovkova3) {
    string dist = "LN";
    double tp = 10.844;
    VectorXd F0{{110, 90}};
    VectorXd sigma{{0.3, 0.2}};
    VectorXd a{{0.7, 0.3}};
    Matrix<double, 2, 2> rho{{1.0, 0.9}, {0.9, 1.0}};
    double K = 104;
    double T = 1;
    double r = 0.03;

    auto res = calculate(T, K, r, a, F0, rho, sigma);
    // std::cout << res << std::endl;
}

TEST(BOP, Borovkova4) {
    string dist = "NLN";
    double tp = 1.958;
    VectorXd F0{{200, 50}};
    VectorXd sigma{{0.1, 0.15}};
    VectorXd a{{-1, 1}};
    Matrix<double, 2, 2> rho{{1.0, 0.8}, {0.8, 1.0}};
    double K = -140;
    double T = 1;
    double r = 0.03;

    auto res = calculate(T, K, r, a, F0, rho, sigma);
    // std::cout << res << std::endl;
}

// TEST(BOP, Borovkova5) {
//     string dist = "NSLN";
//     double tp = 1.958;
//     VectorXd F0{{200, 50}};
//     VectorXd sigma{{0.1, 0.15}};
//     VectorXd a{{-1, 1}};
//     Matrix<double, 2, 2> rho{{1.0, 0.8}, {0.8, 1.0}};
//     double K = -140;
//     double T = 1;
//     double r = 0.03;

//     calculate(T, K, r, dist, a, F0, rho, sigma);
// }

/*
BC = namedtuple("BC", "dist tp F0 sigma a rho K")
borovkova_baskets_1 = [
    BC(
        "NSLN",
        7.759,
        [95, 90, 105],
        [0.2, 0.3, 0.25],
        [1, -0.8, -0.5],
        {(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.8},
        -30,
    ),
    BC(
        "SLN",
        9.026,
        [100, 90, 95],
        [0.25, 0.3, 0.2],
        [0.6, 0.8, -1],
        {(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.8},
        35,
    ),
]
*/
