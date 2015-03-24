#include <alps/params.hpp>
#include <boost/program_options.hpp>

#include "opendf/lattice_traits.hpp"
#include "opendf/df_hubbard.hpp"

namespace po = boost::program_options;
using namespace open_df;

static constexpr int D = 2; // work in 2 dimensions
typedef cubic_traits<D> lattice_t; 
typedef df_hubbard<lattice_t> df_type; 

typedef df_type::vertex_type vertex_type;
typedef df_type::fvertex_type fvertex_type;
typedef df_type::gw_type gw_type;
typedef df_type::gk_type gk_type;
typedef df_type::disp_type disp_type;

alps::params cmdline_params(int argc, char* argv[]);
inline void print_section (const std::string& str);

int main(int argc, char *argv[])
{
    print_section("Atomic limit data");
    alps::params p = cmdline_params(argc, argv); 

    double U = p["U"]; 
    double beta = p["beta"];
    double mu = U/2;
    int nfermionic = p["nfermionic"];
    int nbosonic = p["nbosonic"];
    int plaintext = p["plaintext"];
    std::string output_file = p["output"];
    std::cout << p << std::endl;

    typedef std::complex<double> complex_type;
    typedef typename fmatsubara_grid::point fpoint; 
    typedef typename bmatsubara_grid::point bpoint; 

    // grids
    fmatsubara_grid fgrid(-nfermionic,nfermionic,beta);
    bmatsubara_grid bgrid(-nbosonic + 1,nbosonic,beta);
        
    double mult = beta*U*U/4.;
    typename gw_type::function_type Lambda = [mult,U](std::complex<double> w){return 1. - U*U/4./w/w;};

            
    double vprec = PI/beta/10;
    // 0000 vertex
    vertex_type F_uuuu(std::forward_as_tuple(bgrid,fgrid,fgrid));
    vertex_type::point_function_type VertexF;
    VertexF = [&](bpoint W,fpoint nu1, fpoint nu2)->std::complex<double>{
            double delta_W_0 = is_float_equal(W.value(), 0, vprec);
            double delta_w1_w4 = (nu1.index() == nu2.index());
            std::complex<double> w1 = nu1.value();
            std::complex<double> w2 = nu1.value() + W.value(); 
            std::complex<double> w3 = nu2.value() + W.value(); 
            std::complex<double> w4 = nu2.value();
            return mult * (delta_W_0 - delta_w1_w4) * Lambda(w1) * Lambda(w3);
        };
    F_uuuu.fill(VertexF);
    
    // 0011 vertex
    vertex_type F_uudd(F_uuuu.grids());
    double exp_bU2 = exp(beta * U / 2.0);
    double inv_exp_bU2 = exp(-beta * U / 2.0);
    VertexF = [&](bpoint Wb,fpoint nu1, fpoint nu2)->std::complex<double>{
            std::complex<double> W = Wb;
            std::complex<double> w1 = nu1.value();
            std::complex<double> w2 = nu1.value() + W; 
            std::complex<double> w3 = nu2.value() + W; 
            std::complex<double> w4 = nu2.value();

            double delta_W_0 = is_float_equal(W, 0, vprec);
            double delta_w1_w4 = is_float_equal(w1, w4, vprec);
            double delta_w2_mw3 = is_float_equal(w2, -w3, vprec);

            return - U 
                   - pow<3>(U)/8. * ( w1*w1 + w2*w2 + w3*w3 + w4*w4) / (w1*w2*w3*w4) 
                   + 3.*pow<5>(U)/16. / (w1*w2*w3*w4) 
                   + mult / (1.0 + exp_bU2)     * (2.0 * delta_w2_mw3 + delta_W_0) * Lambda(w2) * Lambda(w3) 
                   - mult / (1.0 + inv_exp_bU2) * (2.0 * delta_w1_w4  + delta_W_0) * Lambda(w1) * Lambda(w3);
        };

    F_uudd.fill(VertexF);

    // magnetic vertex
    vertex_type magnetic_vertex = F_uuuu - F_uudd;
    // density vertex
    vertex_type density_vertex = F_uuuu + F_uudd;

    auto W0 = bgrid.find_nearest(0.0);
    fvertex_type F_upup_static(fgrid,fgrid), F_updn_static(fgrid,fgrid);
    F_upup_static.data() = F_uuuu[W0];
    F_updn_static.data() = F_uudd[W0];

    if (plaintext) { 
        F_upup_static.savetxt("F00_w0.dat");
        F_updn_static.savetxt("F01_w0.dat");
        }

    // local gf
    std::function<complex_type(fpoint)> gw_f = [U](fpoint w) { return 0.5 / (w.value() - U/2.) + 0.5 / (w.value() + U/2.); };
    gw_type gw(fgrid);
    gw.fill(gw_f);

    gw_type delta(gw.grids());
    delta = gw * 2 * double(D);

    std::string top = "/atomic"; 
    std::cout << "Saving data to " << output_file << top << std::endl; 
    alps::hdf5::archive ar(output_file, "w");

    save_grid_object(ar, top + "/magnetic_vertex", magnetic_vertex, plaintext > 1); 
    save_grid_object(ar, top + "/density_vertex", density_vertex, plaintext > 1); 
    save_grid_object(ar, top + "/F00", F_uuuu, plaintext > 1); 
    save_grid_object(ar, top + "/F01", F_uudd, plaintext > 1); 
    save_grid_object(ar, top + "/gw0", gw, plaintext > 0); 
    save_grid_object(ar, top + "/gw1", gw, plaintext > 0); 
    save_grid_object(ar, top + "/delta0", delta, plaintext > 0); 
    save_grid_object(ar, top + "/delta1", delta, plaintext > 0); 
    
}

/// Check that 2 cmd options are not specified at the same time.
void conflicting_options(const po::variables_map& vm, const char* opt1, const char* opt2);

alps::params cmdline_params(int argc, char* argv[])
{
    alps::params p;

	po::options_description generic_opts("generic"), double_opts("double opts"), vec_double_opts("vector<double> opts"), 
		int_opts("int_opts"), string_opts("string_opts"), bool_opts("bool opts");

    generic_opts.add_options()
        ("help",          "help");
    double_opts.add_options()
        ("U",                  po::value<double>()->default_value(16.0),  "value of U")
        ("beta",               po::value<double>()->default_value(1.0),  "value of inverse temperature");
    int_opts.add_options()
        ("nfermionic",             po::value<int>()->default_value(40), "amount of positive fermionic freqs")
        ("nbosonic",         po::value<int>()->default_value(1), "amount of positive bosonic freqs")
        ("plaintext,p",      po::value<int>()->default_value(1), "save additionally to plaintext files (2 = verbose, 1 = save essential, 0 = no plaintext)");
    string_opts.add_options()
        ("output",          po::value<std::string>()->default_value("output.h5"), "output file");

    po::options_description cmdline_opts;
    cmdline_opts.add(double_opts).add(int_opts).add(string_opts).add(generic_opts).add(bool_opts).add(vec_double_opts);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, cmdline_opts, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);

    // Help options
    if (vm.count("help")) { 
        std::cout << "Program options : \n" << cmdline_opts << std::endl; exit(0);  }

    for (auto x : double_opts.options()) p[x->long_name()] = vm[x->long_name()].as<double>();
    for (auto x : int_opts.options()) p[x->long_name()] = vm[x->long_name()].as<int>();
    for (auto x : string_opts.options()) p[x->long_name()] = vm[x->long_name()].as<std::string>();
    for (auto x : bool_opts.options()) p[x->long_name()] = vm[x->long_name()].as<bool>();
    for (auto x : vec_double_opts.options()) p[x->long_name()] = vm[x->long_name()].as<std::vector<double>>();
    
    return p;
}

inline void print_section (const std::string& str)
{
  std::cout << std::string(str.size(),'=') << std::endl;
  std::cout << str << std::endl;
  std::cout << std::string(str.size(),'=') << std::endl;
}

void conflicting_options(const po::variables_map& vm, const char* opt1, const char* opt2)
{
    if (vm.count(opt1) && !vm[opt1].defaulted() && vm.count(opt2) && !vm[opt2].defaulted())
        throw std::logic_error(std::string("Conflicting options '") + opt1 + "' and '" + opt2 + "'.");
}



