/** 
 * This is a simple unoptimized code to evaluate some low-order dual diagrams in FK model
 */

#include <boost/program_options.hpp>
#include <Eigen/Core>
#include <gftools.hpp>
#include "opendf/hdf5.hpp"

namespace po = boost::program_options;
using namespace gftools;

typedef grid_object<std::complex<double>, bmatsubara_grid, fmatsubara_grid, fmatsubara_grid> vertex_type;
typedef grid_object<std::complex<double>, fmatsubara_grid> gw_type;
typedef grid_object<std::complex<double>, fmatsubara_grid, kmesh, kmesh> gk_type;
typedef grid_object<std::complex<double>, kmesh, kmesh> disp_type;

std::array<gw_type,2> read_gw(std::string fname, bool complex_data, bool skip_first_line = false);
std::array<vertex_type, 2> read_vertex(std::string input_file, double beta, bool skip_first_line = false);

int main(int argc, char *argv[])
{
    // parse command line options
    po::options_description desc("FK diagrams evaluator"); 
    desc.add_options()
        ("vertex_file", po::value<std::string>()->default_value("vertexF.dat"), "vertex file")
        ("gw_file", po::value<std::string>()->default_value("gw.dat"), "local gw file")
        ("sigma_file", po::value<std::string>()->default_value("sigma.dat"), "local sigma file")
        ("plaintext,p", po::value<bool>()->default_value(false), "save plaintext data")
        ("help", "produce help message");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << std::endl; exit(0); }

    std::string vertex_file=vm["vertex_file"].as<std::string>();
    std::string gw_file=vm["gw_file"].as<std::string>();
    std::string sigma_file=vm["sigma_file"].as<std::string>();
    bool plain_text = vm["plaintext"].as<bool>();

    std::string outfile_name = "qmc_output.h5";
    std::cout << "Saving data to " << outfile_name << std::endl;
    alps::hdf5::archive ar(outfile_name, "w");

    auto gw_arr = read_gw(gw_file, false, true); 
    save_grid_object(ar, "gw0_input", gw_arr[0]);
    save_grid_object(ar, "gw1_input", gw_arr[1]);
    auto sigma_arr = read_gw(sigma_file, true, false);
    save_grid_object(ar, "sigma0_input", sigma_arr[0]);
    save_grid_object(ar, "sigma1_input", sigma_arr[1]);

    fmatsubara_grid fgrid_gw = gw_arr[0].grid();
    gw_type iw(fgrid_gw); for (auto x : fgrid_gw.points()) iw[x] = x; 
    std::array<gw_type, 2> delta_arr = gftools::tuple_tools::repeater<gw_type, 2>::get_array(gw_type(fgrid_gw));
    double mu = sigma_arr[0][0].real();
    std::cout << "mu = " << mu << std::endl;
    for (int s = 0; s < 2; s++) { 
        delta_arr[s] =  iw + mu - sigma_arr[s] - 1./gw_arr[s];
        save_grid_object(ar, "delta" + std::to_string(s) + "_input", delta_arr[s]);
        }

    double beta = gw_arr[0].grid().beta();
    std::cout << "beta = " << beta << std::endl;
    assert(sigma_arr[0].grid().beta() == gw_arr[0].grid().beta());
    auto vertex_array = read_vertex(vertex_file, beta);
    vertex_type const& F_upup = std::get<0>(vertex_array);
    vertex_type const& F_updn = std::get<1>(vertex_array);

    save_grid_object(ar, "F00", F_upup);
    save_grid_object(ar, "F01", F_updn);

    if (plain_text) { 
        delta_arr[0].savetxt("delta0.dat");
        delta_arr[1].savetxt("delta1.dat");
        gw_arr[0].savetxt("gw0.dat");
        gw_arr[1].savetxt("gw1.dat");
        sigma_arr[0].savetxt("sigma0.dat");
        sigma_arr[1].savetxt("sigma1.dat");
        }

    // prepare df input data
    fmatsubara_grid const& fgrid = F_upup.template grid<1>(); 

    for (int s=0; s<2; s++) { 
        gw_type const& gw_in = gw_arr[s];
        gw_type const& delta_in = delta_arr[s];

        gw_type gw(fgrid), delta(fgrid);

        for (auto x : fgrid.points()) { 
            int n = fgrid.getNumber(x);
            if (n >= 0) { 
                auto p = fgrid_gw.find_nearest(FMatsubara(n, beta));
                gw[x] = gw_in[p];
                delta[x] = delta_in[p];
                }
            else { 
                n = -n - 1;
                auto p = fgrid_gw.find_nearest(FMatsubara(n, beta));
                gw[x] = std::conj(gw_in[p]);
                delta[x] = std::conj(delta_in[p]);
                }
            }
        save_grid_object(ar, "gw" + std::to_string(s), gw);
        save_grid_object(ar, "delta" + std::to_string(s), delta);
        if (plain_text) { 
            gw.savetxt("gw" + std::to_string(s) + "_symm.dat");
            delta.savetxt("delta" + std::to_string(s) + "_symm.dat");
            }
        }
}

std::array<gw_type,2> read_gw(std::string fname, bool complex_data, bool skip_first_line)
{
    std::cout << "Parsing " << fname << std::endl;
    std::ifstream in;
    in.open(fname.c_str());
    if (in.fail()) { ERROR("Couldn't open file " << fname); throw std::logic_error("Couldn't open file " + fname); };
    if (skip_first_line) { std::string comment; std::getline(in, comment); std::cout << "Skipping " << comment << std::endl; } 

    typedef typename gw_type::value_type value_type;
    typedef typename gw_type::arg_tuple arg_tuple;

    std::vector<arg_tuple> grid_vals;
    std::vector<value_type> vals_up, vals_dn;

    while (!in.eof()) {  
        double v, v2=0; 
        in >> v; 
        if (in.eof()) break;
        grid_vals.push_back(I*v);

        in >> v; 
        v2 = v; v=0;
        if (complex_data) { 
            in >> v;
            vals_up.push_back(v2 + v*I);
            }
        else vals_up.push_back(v2*I);

        in >> v; 
        v2 = v; v=0;
        if (complex_data) { 
            in >> v;
            vals_dn.push_back(v2 + v*I);
            }
        else vals_dn.push_back(v2*I);
        }

    typedef typename gw_type::grid_tuple grid_tuple;
    grid_tuple grids = extra::arg_vector_to_grid_tuple<grid_tuple>(grid_vals);
    gw_type out_up(grids), out_dn(grids);
    std::copy(vals_up.begin(), vals_up.end(), out_up.data().data()); 
    std::copy(vals_dn.begin(), vals_dn.end(), out_dn.data().data()); 
    return {{ std::move(out_up), std::move(out_dn) }};
}

std::array<vertex_type, 2> read_vertex(std::string fname, double beta, bool skip_first_line)
{
    std::cout << "Parsing " << fname << std::endl;
    std::ifstream in;
    in.open(fname.c_str());
    if (in.fail()) { ERROR("Couldn't open file " << fname); throw std::logic_error("Couldn't open file " + fname); };
    if (skip_first_line) { std::string comment; std::getline(in, comment); std::cout << "Skipping " << comment << std::endl; } 

    typedef std::tuple<int,int,int> int_tuple; 
    typedef typename vertex_type::value_type value_type;

    std::vector<int_tuple> grid_int_vals;
    std::vector<value_type> vals_upup, vals_updn;

    while (!in.eof()) {  
        int_tuple pts = tuple_tools::read_tuple<int_tuple>(in); // ensure serialize_tuple in savetxt has the same type. Dropping here indices - they're wrong anyway.
        if (in.eof()) break;
        grid_int_vals.push_back(pts);

        value_type v; 
        in >> num_io<value_type>(v); 
        vals_upup.push_back(v);

        in >> num_io<value_type>(v); 
        vals_updn.push_back(v);

        }

    typedef typename vertex_type::arg_tuple arg_tuple;
    std::vector<arg_tuple> grid_vals(grid_int_vals.size());

    for (int i=0; i<grid_int_vals.size(); i++) {  
        int bindex, f1index, f2index; 
        std::tie(bindex, f1index, f2index) = grid_int_vals[i];
        arg_tuple pts = std::forward_as_tuple(BMatsubara(bindex, beta), FMatsubara(f1index, beta), FMatsubara(f2index, beta));
        grid_vals[i] = pts;
        }

    typedef typename vertex_type::grid_tuple grid_tuple;
    grid_tuple grids = extra::arg_vector_to_grid_tuple<grid_tuple>(grid_vals);
    
    vertex_type out_upup(grids), out_updn(grids);
    std::copy(vals_upup.begin(), vals_upup.end(), out_upup.data().data()); 
    std::copy(vals_updn.begin(), vals_updn.end(), out_updn.data().data()); 
    return {{ std::move(out_upup), std::move(out_updn) }};
}


