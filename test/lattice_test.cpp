#include <alps/params.hpp>
#include <gtest/gtest.h>

#include "opendf/lattice_traits.hpp"

using namespace open_df;

template <typename LatticeT> void test_symmetry(LatticeT lattice, int kpts = 32)
{
    kmesh kgrid(kpts);
    typedef typename LatticeT::bz_point bz_point;
    constexpr int D = LatticeT::NDim;

    std::vector<bz_point> all_pts = lattice.getAllBZPoints(kgrid);
    std::map<bz_point, std::vector<bz_point>> symmetric_pts = lattice.getUniqueBZPoints(kgrid);
    double volume = all_pts.size();

    double disp_square = 0.0;
    double disp_sum = 0.0;

    for (std::pair<bz_point, std::vector<bz_point>> bz_pair : symmetric_pts) { 
        bz_point orig = bz_pair.first;
        std::vector<bz_point> symmetric = bz_pair.second;
        double disp_val = lattice.dispersion(orig);
        for (bz_point p : symmetric) { 
            ASSERT_NEAR(disp_val, lattice.dispersion(p), 1e-12);
            }
        disp_sum += disp_val / volume * symmetric.size();
        disp_square += disp_val * disp_val / volume * symmetric.size(); 
        }
    std::cout << "dispersion integral = " << disp_sum << std::endl;
    std::cout << "dispersion^2 integral = " << disp_square << std::endl;

    EXPECT_NEAR(disp_square, lattice.disp_square_sum(), 1e-8);
    EXPECT_NEAR(disp_sum, 0.0, 1e-8);
}

TEST(lattice_test, linear_test) { test_symmetry( cubic_traits<1>(1.0)); }
TEST(lattice_test, square_test) { test_symmetry( cubic_traits<2>(1.0)); }
TEST(lattice_test, cubic_test)  { test_symmetry( cubic_traits<3>(1.0)); }
TEST(lattice_test, cubic4_test) { test_symmetry( cubic_traits<4>(1.0), 14); }
TEST(lattice_test, triangular_test01) { test_symmetry( triangular_traits(1.0, 0.0)); }
TEST(lattice_test, triangular_test02) { test_symmetry( triangular_traits(1.0, 0.317)); }
TEST(lattice_test, square_nnn_test01) { test_symmetry( square_nnn_traits(1.0, 0.0)); }
TEST(lattice_test, square_nnn_test02) { test_symmetry( square_nnn_traits(1.0, 0.317)); }


