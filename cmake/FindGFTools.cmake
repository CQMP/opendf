#  Try to find GFTools. Once done this will define
#  GFTOOLS_FOUND - System has GFTools
#  GFTOOLS_INCLUDE_DIRS - The GFTools include directories
#  GFTOOLS_DEFINITIONS - Compiler switches required for using GFTools

find_package(PkgConfig)
pkg_check_modules(PC_GFTOOLS QUIET gftools)
set(GFTOOLS_DEFINITIONS ${PC_GFTOOLS_CFLAGS_OTHER})
set(gf_inc ${GFTOOLS_INCLUDE_DIR})

find_path(GFTOOLS_INCLUDE_DIR gftools/grid_base.hpp
          HINTS ${PC_GFTOOLS_INCLUDEDIR} ${PC_GFTOOLS_INCLUDE_DIRS} ${gf_inc} 
         )

set(GFTOOLS_INCLUDE_DIRS ${GFTOOLS_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set GFTOOLS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GFTools "No GFTools found" GFTOOLS_INCLUDE_DIR)

mark_as_advanced(GFTOOLS_INCLUDE_DIR GFTOOLS_DEFINITIONS)
