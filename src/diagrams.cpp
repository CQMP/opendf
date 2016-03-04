#include "opendf/diagrams.hpp"

#include <Eigen/LU>
#include <Eigen/Dense>

namespace open_df { 
namespace diagrams { 

template <bool F> matrix_type BetheSalpeter<matrix_type, F>::solve_iterations(size_t n_iter, real_type mix, bool evaluate_only_order_n)
{
    matrix_type V4     = vertex_;
    matrix_type V4_old = vertex_;
    matrix_type V4Chi  = vertex_*bubble_;
    if (verbosity_) INFO_NONEWLINE("\tEvaluating BS self-consistently. Making " << n_iter << " iterations.");
    real_type diffBS = 1.0;
    for (size_t n=0; n<n_iter && diffBS > 1e-8 * double(!evaluate_only_order_n); ++n) { 
        if (verbosity_ > 1) INFO_NONEWLINE("\t\t" << n+1 << "/" << n_iter<< ". ")
        if (fwd)
            V4 = (double(n==n_iter - 1 || !evaluate_only_order_n) * vertex_ + V4Chi*V4_old)*mix + (1.0-mix)*V4_old;
        else 
            V4 = (double(n==n_iter - 1 || !evaluate_only_order_n) * vertex_ - V4_old*V4Chi)*mix + (1.0-mix)*V4_old;
        diffBS = (V4-V4_old).norm();
        if (verbosity_ > 1) INFO("vertex diff = " << diffBS);
        V4_old = V4;
        }
    return std::move(V4);
}

template <bool F> matrix_type BetheSalpeter<matrix_type, F>::solve_inversion()
{
    if (verbosity_ > 0) INFO_NONEWLINE("Running" << ((!fwd)?" inverse ":" ") << "matrix BS equation...");
    size_t size = vertex_.rows(); 
    matrix_type V4Chi = fwd ? matrix_type(matrix_type::Identity(size,size) - vertex_*bubble_) : matrix_type(matrix_type::Identity(size,size) + bubble_*vertex_);

    Eigen::PartialPivLU<matrix_type> Solver(V4Chi);
    det_ = Solver.determinant(); 

    assert(is_float_equal(det_, V4Chi.determinant(), 1e-6));

    if (std::abs(std::imag(det_))>1e-2 * std::abs(std::real(det_))) { ERROR("Determinant : " << det_); throw (std::logic_error("Complex determinant in BS. Exiting.")); };
    if ((std::real(det_))<1e-2) INFO3("Determinant : " << det_);

    if (std::real(det_) < std::numeric_limits<real_type>::epsilon()) {
        ERROR("Can't solve Bethe-Salpeter equation by inversion");
        return std::move(V4Chi);
    }

    V4Chi = fwd ? matrix_type(Solver.solve(vertex_)) : matrix_type(vertex_*Solver.inverse());
                //V4Chi=vertex_*Solver.inverse();
    if (verbosity_ > 0) INFO("done.");
    return std::move(V4Chi);
}

template <bool F> matrix_type BetheSalpeter<matrix_type, F>::solve(bool eval_iterations, size_t n_iter, real_type mix, bool evaluate_only_order_n)
{
    matrix_type out;
    if (!eval_iterations) out = this->solve_inversion(); 
    eval_iterations = eval_iterations || det_.real() < std::numeric_limits<double>::epsilon(); 
    if (eval_iterations) out = this->solve_iterations(n_iter, mix, evaluate_only_order_n);
    return std::move(out);     
}

template class BetheSalpeter<matrix_type, true>;
template class BetheSalpeter<matrix_type, false>;

std::complex<double> max_eval(const matrix_type &bubble_, const matrix_type &vertex_)
{
    auto evals = Eigen::ComplexEigenSolver<matrix_type>(vertex_*bubble_).eigenvalues();
    std::sort(&evals[0], &evals[evals.size()], [](std::complex<double> x, std::complex<double> y){return x.real() > y.real(); });
    return evals[evals.size() - 1];
} 

} // end of namespace diagrams
} // end of namespace open_df
