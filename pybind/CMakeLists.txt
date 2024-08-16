set(BOP_MODULE basket_option_pricer)

file(GLOB_RECURSE PYXMOLPP_SOURCES
    basket_option_pricer/**.cpp
    basket_option_pricer/**.h
)
pybind11_add_module(${BOP_MODULE} ${PYXMOLPP_SOURCES})
target_include_directories(${BOP_MODULE} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/basket_option_pricer PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(${BOP_MODULE} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(${BOP_MODULE} PRIVATE ${BOP_LIB_STATIC})

# set(VENV_DIR /home/blair/projects/basket_option_pricer/.venv)
