/** 
 * This is a simple unoptimized code to evaluate some low-order dual diagrams in FK model
 * input : 
 *  --order : order of the diagram
 *  --vertex_file : full impurity vertex. default : "gamma4.dat"
 *  --gd0_file : bare dual Green's function. default : "gd0_k.dat"
 * output:
 *  - dual self-energy "sigma_wk.dat"
 *  - cut of dual self-energy at first Matsubara freq
 *  - k-dependence of dual bubbles (summer over Matsubara freqs). 
 */

#include <boost/program_options.hpp>
#include <gftools.hpp>
#include <Eigen/Core>

namespace po = boost::program_options;
using namespace gftools;

int main(int argc, char *argv[])
{
    // parse command line options
    po::options_description desc("FK diagrams evaluator"); 
    desc.add_options()
        ("order,n", po::value<int>()->default_value(2), "order of diagrams")
        ("vertex_file", po::value<std::string>()->default_value("gamma4.dat"), "vertex file")
        ("gd0_file", po::value<std::string>()->default_value("gd0_k.dat"), "gd0 file")
        ("help", "produce help message");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << std::endl; exit(0); }

    std::string vertex_file=vm["vertex_file"].as<std::string>();
    std::string gd0_file=vm["gd0_file"].as<std::string>();
    int diagram_order = vm["order"].as<int>(); 
    // we will use diagram_order as amount of iterations in vertices. 1 iteration = 2nd order, etc. 
    diagram_order-=(diagram_order>0);

    static constexpr int NDim = 2; // work in 2 dimensions

    typedef grid_object<std::complex<double>, fmatsubara_grid, fmatsubara_grid> vertex_type;
    typedef grid_object<std::complex<double>, fmatsubara_grid> gw_type;
    typedef grid_object<std::complex<double>, fmatsubara_grid, kmesh, kmesh> gk_type;
    typedef grid_object<std::complex<double>, kmesh, kmesh> ek_type;
    typedef Eigen::MatrixXcd matrix_type; 
    
    // load vertex
    std::cout << "Loading vertex from " << vertex_file << std::endl;
    vertex_type gamma4 = loadtxt<vertex_type>(vertex_file);
    // define full vertex (will be evaluated below)
    vertex_type full_vertex(gamma4);
    // make a matrix from the vertex
    matrix_type gamma4_matrix = gamma4.data().as_matrix();
    matrix_type full_vertex_matrix(gamma4_matrix);
    // define frequency grid for the evaluation and extract temperature
    const fmatsubara_grid& fgrid(gamma4.grid());
    double beta = fgrid.beta(); 
    double T = 1.0/beta;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "T = " << T << std::endl;
    // load bare dual gf
    std::cout << "Loading gd0 from " << gd0_file << std::endl;
    gk_type gd0 = loadtxt<gk_type>(gd0_file);
    // check that we are consistent - 2 frequency grids are the same
    if (gd0.template grid<0>() != fgrid) throw std::logic_error("matsubara grid mismatch");
    
    // define mesh in Brilloin zone (BZ)
    const kmesh& kgrid = gd0.template grid<1>();
    int kpts = kgrid.size();
    int totalkpts = std::pow(kpts, NDim);
    std::cout << "kmesh : " << kpts << " points " << std::endl; 
    std::cout << "total pts in BZ = " << totalkpts << std::endl; 
    std::cout << "kmesh : " << kgrid << std::endl;
    // Initalize dual self-energy on the same grids as gd0
    gk_type sigma_dual(gd0.grids());
    sigma_dual = 0.0;
    ek_type bare_bubbles(kgrid,kgrid);

    // now loop through the BZ (no irreducible part optimization)
    for (kmesh::point q1 : kgrid.points()) { 
        for (kmesh::point q2 : kgrid.points()) { 
            std::cout << "[" << q1.index()*kpts + q2.index()+1 << "/" << totalkpts <<  "]; q = {" <<  q1.value() << " " << q2.value() << "}" << std::endl;
            // Evaluate -T \sum_k G_{w,k} G_{w,k+q}
            // Shift dual g in k-space and make no frequency shift. 
            // note : this operation is typically optimized via an fft. 
            gk_type gd0_shift = gd0.shift(std::make_tuple(0.0,q1,q2)); 
            // obtain a bubble 
            gk_type bubble_wk = -T * gd0 * gd0_shift;
            // perform sum over k    
            gw_type dual_bubble(fgrid);
            for (auto w : fgrid.points()) { dual_bubble[w] = bubble_wk[w].sum() / double(totalkpts); }
            // save the bubble for output 
            bare_bubbles(q1, q2) = dual_bubble.sum();
            // construct a diagonal matrix (in frequency space from the bubble)
            matrix_type dual_bubble_matrix = dual_bubble.data().as_diagonal_matrix(); 
            // refresh vertex
            full_vertex_matrix = gamma4_matrix;
            // IMPORTANT : Evaluate dual diagram
            for (int n=0; n<diagram_order; n++) 
                full_vertex_matrix= gamma4_matrix * dual_bubble_matrix * full_vertex_matrix; 
            // update self-energy
            for (auto w : fgrid.points())
                sigma_dual[w.index()] += T* full_vertex_matrix(w.index(), w.index()) * gd0_shift[w.index()] / double(totalkpts);
        }
    }
    // output
    // save sigma
    sigma_dual.savetxt("sigma_wk.dat");
    // save sigma at first matsubara
    auto w0 = fgrid.find_nearest(I*PI/beta);
    ek_type sigma_w0(std::forward_as_tuple(kgrid,kgrid),sigma_dual[w0]);
    sigma_w0.savetxt("sigma_w0.dat");
    // save bare bubbles
    bare_bubbles.savetxt("db0.dat");
}

