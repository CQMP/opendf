include_directories(..)

include(EnableGtests) #defined in common/cmake

set (tests
    atomic_limit_test
    lattice_test
 )

foreach (test ${tests})
    add_gtest(${test})
    target_link_libraries(${test} ${opendf_name})
    if (NOT GFTOOLS_FOUND)
        add_dependencies(${test} gftools)
    endif()
endforeach(test)


