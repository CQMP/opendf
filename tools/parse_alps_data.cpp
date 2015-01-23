/** 
 * This is a simple unoptimized code to evaluate some low-order dual diagrams in FK model
 */

#include <boost/program_options.hpp>
#include <alps/params.hpp>
#include <Eigen/Core>
#include <gftools.hpp>
#include "opendf/hdf5.hpp"

namespace po = boost::program_options;
using namespace gftools;

typedef grid_object<std::complex<double>, bmatsubara_grid, fmatsubara_grid, fmatsubara_grid> vertex_type;
typedef grid_object<std::complex<double>, fmatsubara_grid, fmatsubara_grid> fvertex_type;
typedef grid_object<std::complex<double>, fmatsubara_grid> gw_type;
typedef grid_object<std::complex<double>, fmatsubara_grid, kmesh, kmesh> gk_type;
typedef grid_object<std::complex<double>, kmesh, kmesh> disp_type;

alps::params cmdline_params(int argc, char* argv[]);
std::array<gw_type,2> read_gw(std::string fname, bool complex_data, bool skip_first_line = false);
std::array<vertex_type, 2> read_vertex(std::string input_file, double beta, bool skip_first_line = false);

int main(int argc, char *argv[])
{
    
    alps::params p = cmdline_params(argc, argv); 

    std::string vertex_file=p["vertex_file"];
    std::string gw_file=p["gw_file"];
    std::string sigma_file=p["sigma_file"];
    int plaintext = p["plaintext"];

    std::string outfile_name = "qmc_output.h5";
    std::string top = "/dmft";
    std::string input = "/dmft_input";
    std::cout << "Saving input data to " << outfile_name << input << std::endl;
    alps::hdf5::archive ar(outfile_name, "w");

    auto gw_arr = read_gw(gw_file, false, true); 
    save_grid_object(ar, input + "/gw0_input", gw_arr[0]);
    save_grid_object(ar, input + "/gw1_input", gw_arr[1]);
    auto sigma_arr = read_gw(sigma_file, true, false);
    save_grid_object(ar, input + "/sigma0_input", sigma_arr[0]);
    save_grid_object(ar, input + "/sigma1_input", sigma_arr[1]);

    fmatsubara_grid fgrid_gw = gw_arr[0].grid();
    gw_type iw(fgrid_gw); for (auto x : fgrid_gw.points()) iw[x] = x; 
    std::array<gw_type, 2> delta_arr = gftools::tuple_tools::repeater<gw_type, 2>::get_array(gw_type(fgrid_gw));
    double mu = p["mu_defaulted"].cast<bool>()?sigma_arr[0][0].real():p["mu"].cast<double>();
    std::cout << "mu = " << mu << std::endl;
    for (int s = 0; s < 2; s++) { 
        delta_arr[s] =  iw + mu - sigma_arr[s] - 1./gw_arr[s];
        save_grid_object(ar, input + "/delta" + std::to_string(s) + "_input", delta_arr[s]);
        }

    double beta = gw_arr[0].grid().beta();
    std::cout << "beta = " << beta << std::endl;
    assert(sigma_arr[0].grid().beta() == gw_arr[0].grid().beta());
    auto input_vertex_array = read_vertex(vertex_file, beta);
    // There is a minus sign between DF and DGA notations :-(
    vertex_type const& F_upup_input = std::get<0>(input_vertex_array)*(-1.);
    vertex_type const& F_updn_input = std::get<1>(input_vertex_array)*(-1.);


    bmatsubara_grid const& bgrid_input = F_upup_input.template grid<0>();
    fmatsubara_grid const& fgrid_input = F_upup_input.template grid<1>(); 
    save_grid_object(ar, input + "/F00_input", F_upup_input);
    save_grid_object(ar, input + "/F01_input", F_updn_input);

    // now move on to export DF-ready data
    std::cout << "Saving data to " << outfile_name << top << std::endl;
    int wfmax = std::min(p["nfermionic"].cast<int>(), fgrid_input.max_n() + 1);
    int wbmax = std::min(p["nbosonic"].cast<int>(), bgrid_input.max_n() + 1);
    fmatsubara_grid fgrid(-wfmax, wfmax, beta);
    bmatsubara_grid bgrid(-wbmax+1, wbmax, beta);
        
    vertex_type F_upup(std::make_tuple(bgrid,fgrid,fgrid));
    F_upup.copy_interpolate(F_upup_input);
    vertex_type F_updn(std::make_tuple(bgrid,fgrid,fgrid));
    F_updn.copy_interpolate(F_updn_input);

    save_grid_object(ar, top + "/F00", F_upup, plaintext > 1);
    save_grid_object(ar, top + "/F01", F_updn, plaintext > 1);

    auto W0 = bgrid.find_nearest(0.0);
    fvertex_type F_upup_static(fgrid,fgrid), F_updn_static(fgrid,fgrid);
    F_upup_static.data() = F_upup[W0];
    F_updn_static.data() = F_updn[W0];

    if (plaintext) { 
        delta_arr[0].savetxt("delta0.dat");
        delta_arr[1].savetxt("delta1.dat");
        gw_arr[0].savetxt("gw0.dat");
        gw_arr[1].savetxt("gw1.dat");
        sigma_arr[0].savetxt("sigma0.dat");
        sigma_arr[1].savetxt("sigma1.dat");
        F_upup_static.savetxt("F00_w0.dat");
        F_updn_static.savetxt("F01_w0.dat");
        }

    // prepare df input data

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
        save_grid_object(ar, top + "/gw" + std::to_string(s), gw);
        save_grid_object(ar, top + "/delta" + std::to_string(s), delta);
        if (plaintext) { 
            gw.savetxt("gw" + std::to_string(s) + "_symm.dat");
            delta.savetxt("delta" + std::to_string(s) + "_symm.dat");
            }
        }

        ar[top + "/parameters"] << p;
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
        grid_vals.push_back(std::make_tuple(I*v));

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

alps::params cmdline_params(int argc, char* argv[])
{
    alps::params p;

	po::options_description generic_opts("generic"), double_opts("double opts"), vec_double_opts("vector<double> opts"), 
		int_opts("int_opts"), string_opts("string_opts"), bool_opts("bool opts");

    generic_opts.add_options()
        ("help",          "help");
    double_opts.add_options()
        ("mu", po::value<double>()->default_value(0.0), "chemical potential (measured from half-filling level)");
    int_opts.add_options()
        ("plaintext,p",      po::value<int>()->default_value(0), "save additionally to plaintext files")
        ("nbosonic",      po::value<int>()->default_value(1024), "max number of non-negative bosonic Matsubara frequencies")
        ("nfermionic",      po::value<int>()->default_value(1024), "max number of positive fermionic Matsubara frequencies");
    string_opts.add_options()
        ("vertex_file", po::value<std::string>()->default_value("vertexF.dat"), "vertex file")
        ("gw_file", po::value<std::string>()->default_value("gw.dat"), "local gw file")
        ("sigma_file", po::value<std::string>()->default_value("sigma.dat"), "local sigma file");

    po::options_description cmdline_opts;
    cmdline_opts.add(double_opts).add(int_opts).add(string_opts).add(generic_opts).add(bool_opts).add(vec_double_opts);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, cmdline_opts, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);

    // Help options
    if (vm.count("help")) { 
        std::cout << "Program options : \n" << cmdline_opts << std::endl; exit(0);  }
    
    p["mu_defaulted"]=vm["mu"].defaulted();

    for (auto x : double_opts.options()) p[x->long_name()] = vm[x->long_name()].as<double>();
    for (auto x : int_opts.options()) p[x->long_name()] = vm[x->long_name()].as<int>();
    for (auto x : string_opts.options()) p[x->long_name()] = vm[x->long_name()].as<std::string>();
    for (auto x : bool_opts.options()) p[x->long_name()] = vm[x->long_name()].as<bool>();
    for (auto x : vec_double_opts.options()) p[x->long_name()] = vm[x->long_name()].as<std::vector<double>>();
    
    return p;
}

