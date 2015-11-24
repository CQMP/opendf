#fk_diagrams 
set (execs 
hub_df
)


foreach (df_lattice ${df_lattices})
    message(STATUS "DF: ${df_lattice}")
    foreach(exec_name ${execs})
        set(exec ${exec_name}_${df_lattice})
        add_executable(${exec} ${exec_name}.cpp)
        set_target_properties(${exec} PROPERTIES COMPILE_DEFINITIONS "LATTICE_${df_lattice}" POSITION_INDEPENDENT_CODE ON) 
        target_link_libraries(${exec} ${opendf_name})
        if (NOT GFTOOLS_FOUND)
            add_dependencies(${exec} gftools)
        endif()
        install (TARGETS ${exec} DESTINATION bin )
    endforeach(exec_name)

    if (OPENDF_HAS_PYTHON) 
        # build python module
        foreach(exec_name ${execs})
            set(exec ${exec_name}_${df_lattice})
            add_library(py${exec} MODULE ${exec_name}.cpp)
            set_target_properties(py${exec} PROPERTIES PREFIX "")
            #add_library(py${exec} ALIAS py${exec})
            target_link_libraries(py${exec} ${opendf_name})
            target_compile_definitions(py${exec} PUBLIC -DBUILD_PYTHON_MODULE -DLATTICE_${df_lattice})
            install ( TARGETS py${exec} DESTINATION lib/python )
        endforeach()
    endif (OPENDF_HAS_PYTHON) 

endforeach(df_lattice)
