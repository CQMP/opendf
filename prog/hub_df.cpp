#include <boost/program_options.hpp>
#include <boost/mpi.hpp>
#include <chrono>
#include <alps/params.hpp>

#include <opendf/lattice_traits.hpp>
#include <opendf/df.hpp>

#include "data_save.hpp"

namespace po = boost::program_options;
using namespace open_df;
using namespace std::chrono;

static constexpr int D = 2; // work in 2 dimensions
typedef cubic_traits<D> lattice_t; 
typedef df_hubbard<lattice_t> df_type; 

typedef df_type::vertex_type vertex_type;
typedef df_type::gw_type gw_type;
typedef df_type::gk_type gk_type;
typedef df_type::disp_type disp_type;


inline void print_section (const std::string& str)
{
  std::cout << std::string(str.size(),'=') << std::endl;
  std::cout << str << std::endl;
  std::cout << std::string(str.size(),'=') << std::endl;
}

void run(alps::params p)
{
    boost::mpi::communicator comm;
    print_section("DF ladder in " + std::to_string(D) + " dimensions.");
    // read input data
    std::string input_name = p["input"] | "qmc_output.h5";
    std::string top = p["inp_section"] | "";
    std::cout << "Reading data from " << input_name << " /" << top << std::endl;

    alps::hdf5::archive ar(input_name, "r");

    std::array<vertex_type, 2> vertex_input = {{ 
        load_grid_object<vertex_type>(ar, top + "/F00"), 
        load_grid_object<vertex_type>(ar, top + "/F01") }}; 

    std::array<gw_type, 2> gw_arr = {{ 
        load_grid_object<gw_type>(ar, top + "/gw0"), 
        load_grid_object<gw_type>(ar, top + "/gw1") }} ;

    std::array<gw_type, 2> delta_arr = {{ 
        load_grid_object<gw_type>(ar, top + "/delta0"), 
        load_grid_object<gw_type>(ar, top + "/delta1") }} ;

    // fix spin symmerty
    if (gw_arr[0].diff(gw_arr[1]) > 1e-8) 
        throw std::logic_error("Spin assymetry is not implemented.");

    gw_type const& gw = gw_arr[0];
    gw_type const& Delta = delta_arr[0]; 

    vertex_type density_vertex = vertex_input[0] + vertex_input[1];
    vertex_type magnetic_vertex = vertex_input[0] - vertex_input[1];

    if (gw.grid() != vertex_input[0].template grid<1>()) 
        throw std::logic_error("Green's function and vertex are defined on different grids. Exiting.");

    // parameters
    double hopping_t = p["hopping"] | 1.0;
    double beta = gw.grid().beta();
    double T = 1.0/beta;
    int kpts = p["kpts"];
    p["beta"] = beta;

    print_section("parameters");
    std::cout << p << std::flush;
    std::cout << "temperature = " << T << std::endl;
    std::cout << std::endl;

    // create a grid over Brilloin zone for evaluation 
    kmesh kgrid(kpts);
    
    // define lattice
    lattice_t lattice(hopping_t);
    // get dispersion
    disp_type disp(std::forward_as_tuple(kgrid, kgrid));
    disp.fill(lattice.get_dispersion());  
        
    // construct a df run 
    df_hubbard<cubic_traits<2>> DF(gw, Delta, lattice, kgrid, density_vertex, magnetic_vertex);
    // run df
    steady_clock::time_point start, end;
    start = steady_clock::now();
    gw_type delta_upd = DF(p);
    end = steady_clock::now();

    if (!comm.rank()) { 

        std::cout << "Calculation lasted : " 
            << duration_cast<hours>(end-start).count() << "h " 
            << duration_cast<minutes>(end-start).count()%60 << "m " 
            << duration_cast<seconds>(end-start).count()%60 << "s " 
            << duration_cast<milliseconds>(end-start).count()%1000 << "ms " 
            << std::endl;
        p["run_time"] = duration_cast<seconds>(end-start).count();

        save_data(DF, p); 
        }
}


//
// build command line executable
// 

#ifndef BUILD_PYTHON_MODULE
alps::params cmdline_params(int argc, char* argv[]);
void conflicting_options(const po::variables_map& vm, const char* opt1, const char* opt2);

int main(int argc, char *argv[])
{
    boost::mpi::environment env(argc, argv);

    alps::params p = cmdline_params(argc, argv); 
    try { run(p); }
    catch (std::exception &e) { std::cerr << e.what() << std::endl; exit(1); };
    
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
        ("df_sc_mix",       po::value<double>()->default_value(1.0),  "mixing between df iterations")
        ("df_sc_cutoff",    po::value<double>()->default_value(1e-7), "cutoff for dual iterations")
        ("bs_mix",          po::value<double>()->default_value(1.0),  "mixing between df iterations")
        ("hopping",          po::value<double>()->default_value(1.0),  "hopping on a lattice");
    int_opts.add_options()
        ("kpts",                    po::value<int>()->default_value(16), "number of points on a single axis in Brilloin zone")
        ("df_sc_iter",       po::value<int>()->default_value(1000), "maximum df iterations")
        ("nbosonic",           po::value<int>()->default_value(10),  "amount of bosonic freqs to use (reduced also by the amount of freqs in the vertex")
        ("n_bs_iter",               po::value<int>()->default_value(100), "amount of self-consistent iterations in BS (with eval_bs_sc = 1)")
        ("plaintext,p",      po::value<int>()->default_value(1), "save additionally to plaintext files (2 = verbose, 1 = save essential, 0 = no plaintext)");
    string_opts.add_options()
        ("input",           po::value<std::string>()->default_value("output.h5"), "input file with vertices and gf")
        ("inp_section",     po::value<std::string>()->default_value("dmft"), "input section (default : 'dmft')")
        ("output",          po::value<std::string>()->default_value("output.h5"), "output file");
    bool_opts.add_options()
        ("resume", po::value<bool>()->default_value(1), "try resuming calculation")
        ("update_df_mixing", po::value<bool>()->default_value(1), "update mixing of dual gf for better accuracy")
        ("eval_bs_sc", po::value<bool>()->default_value(0), "evaluate Bethe-Salpeter equation self-consistently");

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
#endif 

void conflicting_options(const po::variables_map& vm, const char* opt1, const char* opt2)
{
    if (vm.count(opt1) && !vm[opt1].defaulted() && vm.count(opt2) && !vm[opt2].defaulted())
        throw std::logic_error(std::string("Conflicting options '") + opt1 + "' and '" + opt2 + "'.");
}


//
// build python module
//

#ifdef BUILD_PYTHON_MODULE
#include <boost/python.hpp>
//compile it as a python module (requires boost::python library)

void solve(boost::python::dict py_parms)
{
    //alps::params p(parms_);
    //std::string output_file = boost::lexical_cast<std::string>(parms["BASENAME"]|"results")+std::string(".out.h5");
    alps::params p;
    #define PYCONV(x, T) p[x] = boost::python::extract<T>(py_parms.attr(x));
    PYCONV("df_sc_mix", double)
    PYCONV("df_sc_cutoff", double)
    PYCONV("bs_mix", double)
    PYCONV("hopping", double)
    
    PYCONV("kpts", int);
    PYCONV("df_sc_iter", int);
    PYCONV("nbosonic", int);
    PYCONV("n_bs_iter", int);
    PYCONV("plaintext", int);

    PYCONV("input", std::string);
    PYCONV("output", std::string);

    PYCONV("update_df_mixing", bool);
    PYCONV("eval_bs_sc", bool);
    PYCONV("resume", bool);
    #undef PYCONV
    run(p);
};

    BOOST_PYTHON_MODULE(libpyhub_df)
    {
        using namespace boost::python;
        def("solve",&solve); //define python-callable run method
    };
#endif


