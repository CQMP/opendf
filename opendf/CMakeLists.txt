include_directories (.)

configure_file(${CMAKE_SOURCE_DIR}/opendf/config.hpp.in ${CMAKE_BINARY_DIR}/opendf/config.hpp)

set (df_src 
    diagrams.hpp diagrams.cpp
    df_base.hpp df_base.cpp 
    df_hubbard.hpp df_hubbard.cpp 
    )

#add_library(${opendf_name} SHARED ${df_src})
add_library(${opendf_name} STATIC ${df_src})

install(TARGETS ${opendf_name}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    )

install(DIRECTORY . DESTINATION include/opendf/
          FILES_MATCHING PATTERN "*.h *.hpp")
