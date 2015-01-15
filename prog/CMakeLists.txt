
#fk_diagrams 
set (execs 
hub_diagrams
)

include_directories(${CMAKE_SOURCE_DIR}/src)

foreach(exec ${execs})
    add_executable(${exec} ${exec}.cpp)
    target_link_libraries(${exec} open_df${DF_POSTFIX})
    if (NOT GFTOOLS_FOUND)
        add_dependencies(${exec} gftools)
    endif()
    install ( TARGETS ${exec} DESTINATION bin )
endforeach()


