include(ExternalProject)

find_package(Git)

ExternalProject_Add(gftools
    GIT_REPOSITORY https://github.com/aeantipov/gftools.git
    GIT_TAG main
    TIMEOUT 10
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    TEST_COMMAND ""
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
)
    
ExternalProject_Get_Property(gftools source_dir)
set(GFTOOLS_INCLUDE_DIR "${source_dir}")

