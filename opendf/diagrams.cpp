#include "diagrams.hpp"

#include <Eigen/LU>
#include <Eigen/Dense>

namespace open_df { 
namespace diagrams { 

matrix_type BS(const matrix_type &bubble_, const matrix_type &vertex_, bool forward, bool eval_SC, size_t n_iter, real_type mix,  bool evaluate_only_order_n)
{
    INFO_NONEWLINE("Running" << ((!forward)?" inverse ":" ") << "matrix BS equation...");
    size_t size = vertex_.rows(); 

    matrix_type V4Chi;
    if (forward)
        V4Chi = matrix_type::Identity(size,size) - vertex_*bubble_;
    else
        V4Chi = matrix_type::Identity(size,size) + bubble_*vertex_;
    //auto ((vertex_*bubble_).eigenvalues())

    Eigen::PartialPivLU<matrix_type> Solver(V4Chi);
    std::complex<double> det = Solver.determinant(); 

    assert(is_float_equal(det, V4Chi.determinant(), 1e-6));

    if (std::abs(std::imag(det))>1e-2 * std::abs(std::real(det))) { ERROR("Determinant : " << det); throw (std::logic_error("Complex determinant in BS. Exiting.")); };
    if ((std::real(det))<1e-2) INFO3("Determinant : " << det);

    if (!eval_SC && std::real(det)>std::numeric_limits<real_type>::epsilon()) {
        try {
            if (forward) {
                V4Chi = Solver.solve(vertex_);
                //V4Chi = V4Chi.inverse()*vertex_; 
                }
            else
                V4Chi=vertex_*V4Chi.inverse();
            INFO("done.");
            return V4Chi;
        }
        catch (std::exception &e) {
            ERROR("Couldn't invert the vertex");
        }
    }; // From here solver by iterations
    auto V4 = vertex_;
    auto V4_old = vertex_;
    V4Chi=vertex_*bubble_;
    INFO_NONEWLINE("\tEvaluating BS self-consistently. Making " << n_iter << " iterations.");
    real_type diffBS = 1.0;
    for (size_t n=0; n<n_iter && diffBS > 1e-8 * double(!evaluate_only_order_n); ++n) { 
        INFO_NONEWLINE("\t\t" << n+1 << "/" << n_iter<< ". ")
        if (forward)
            V4 = (double(n==n_iter - 1 || !evaluate_only_order_n) * vertex_ + V4Chi*V4_old)*mix + (1.0-mix)*V4_old;
        else 
            V4 = (double(n==n_iter - 1 || !evaluate_only_order_n) * vertex_ - V4_old*V4Chi)*mix + (1.0-mix)*V4_old;
        diffBS = (V4-V4_old).norm();
        INFO("vertex diff = " << diffBS);
        V4_old = V4;
        }
    return V4;
}

std::complex<double> max_eval(const matrix_type &bubble_, const matrix_type &vertex_)
{
    auto evals = Eigen::ComplexEigenSolver<matrix_type>(vertex_*bubble_).eigenvalues();
    std::sort(&evals[0], &evals[evals.size()], [](std::complex<double> x, std::complex<double> y){return x.real() > y.real(); });
    return evals[evals.size() - 1];
} 

} // end of namespace diagrams
} // end of namespace open_df
