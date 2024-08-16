#include <cmath>
#include <numbers>

using std::erfc, std::exp, std::sqrt, std::log;
using std::numbers::inv_sqrtpi_v, std::numbers::sqrt2_v;

static const double inv_sqrt2_v = 1 / sqrt2_v<double>;
static const double inv_sqrt2pi_v = inv_sqrtpi_v<double> * inv_sqrt2_v;


double Phi(double x)
{
   return 0.5 * erfc(-x * inv_sqrt2_v);
}

double phi(double x)
{
    return inv_sqrt2pi_v * exp(-0.5f * x * x);
}
