#include "diagrams.hpp"

#include <Eigen/LU>
#include <Eigen/Dense>

namespace open_df { 
namespace diagrams { 

matrix_type BS(const matrix_type &Chi0, const matrix_type &IrrVertex4, bool forward, bool eval_SC, size_t n_iter, real_type mix,  bool evaluate_only_order_n)
{
    INFO_NONEWLINE("Running" << ((!forward)?" inverse ":" ") << "matrix BS equation...");
    size_t size = IrrVertex4.rows(); 

    matrix_type V4Chi;
    if (forward)
        V4Chi = matrix_type::Identity(size,size) - IrrVertex4*Chi0;
    else
        V4Chi = matrix_type::Identity(size,size) + Chi0*IrrVertex4;
    //auto ((IrrVertex4*Chi0).eigenvalues())
    auto D1 = V4Chi.determinant();
    if (std::abs(std::imag(D1))>1e-2 * std::abs(std::real(D1))) { ERROR("Determinant : " << D1); throw (std::logic_error("Complex determinant in BS. Exiting.")); };
    if (std::abs(std::real(D1))<1e-2) INFO3("Determinant : " << D1);

    if (!eval_SC && std::real(D1)>std::numeric_limits<real_type>::epsilon()) {
        try {
            if (forward) {
                V4Chi = V4Chi.colPivHouseholderQr().solve(IrrVertex4);
                //V4Chi = V4Chi.inverse()*IrrVertex4; 
                }
            else
                V4Chi=IrrVertex4*V4Chi.inverse();
            INFO("done.");
            return V4Chi;
        }
        catch (std::exception &e) {
            ERROR("Couldn't invert the vertex");
        }
    }; // From here solver by iterations
    auto V4 = IrrVertex4;
    auto V4_old = IrrVertex4;
    V4Chi=IrrVertex4*Chi0;
    INFO_NONEWLINE("\tEvaluating BS self-consistently. Making " << n_iter << " iterations.");
    real_type diffBS = 1.0;
    for (size_t n=0; n<n_iter && diffBS > 1e-8 * double(!evaluate_only_order_n); ++n) { 
        INFO_NONEWLINE("\t\t" << n+1 << "/" << n_iter<< ". ")
        if (forward)
            V4 = (double(n==n_iter - 1 || !evaluate_only_order_n) * IrrVertex4 + V4Chi*V4_old)*mix + (1.0-mix)*V4_old;
        else 
            V4 = (double(n==n_iter - 1 || !evaluate_only_order_n) * IrrVertex4 - V4_old*V4Chi)*mix + (1.0-mix)*V4_old;
        diffBS = (V4-V4_old).norm();
        INFO("vertex diff = " << diffBS);
        V4_old = V4;
        }
    return V4;
}

} // end of namespace diagrams
} // end of namespace open_df
