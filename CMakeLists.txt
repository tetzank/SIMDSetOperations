cmake_minimum_required(VERSION 2.8)

# build type defaults to debug
if(NOT DEFINED CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
endif(NOT DEFINED CMAKE_BUILD_TYPE)
# set default compiler flags to enable C++17
if(NOT DEFINED CMAKE_CXX_FLAGS)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -std=c++17 -march=native" CACHE STRING "Flags used by the compiler during all build types.")
endif(NOT DEFINED CMAKE_CXX_FLAGS)
# enable some optimization even in debug build
if(NOT DEFINED CMAKE_CXX_FLAGS_DEBUG)
	set(CMAKE_CXX_FLAGS_DEBUG "-g -Og" CACHE STRING "Flags used by the compiler during debug builds.")
endif(NOT DEFINED CMAKE_CXX_FLAGS_DEBUG)

project(SIMDSetOperations CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")


find_package(OpenMP REQUIRED)
find_package(GBenchmark)

include(${CMAKE_SOURCE_DIR}/cmake/iaca.cmake)


option(DISABLE_ASM "disable variants with inline assembly" OFF)


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/projectconfig.h.in" "${CMAKE_CURRENT_BINARY_DIR}/projectconfig.h")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${OpenMP_CXX_FLAGS})


add_executable(genlists genLists.cpp)

add_executable(set_intersection set_intersection.cpp)
set_target_properties(set_intersection PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})

add_executable(set_union set_union.cpp)
set_target_properties(set_union PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})

add_executable(set_difference set_difference.cpp)
set_target_properties(set_difference PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})

add_executable(merge merge.cpp)
set_target_properties(merge PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})

add_executable(tests tests.cpp)
set_target_properties(tests PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})

if(${GBENCHMARK_FOUND})
	add_definitions(${GBENCHMARK_INCLUDE_DIR})

	add_executable(benchmark_intersection benchmark_intersection.cpp)
	target_link_libraries(benchmark_intersection ${GBENCHMARK_LIBRARIES})
	set_target_properties(benchmark_intersection PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})

	add_executable(benchmark_union benchmark_union.cpp)
	target_link_libraries(benchmark_union ${GBENCHMARK_LIBRARIES})
	set_target_properties(benchmark_union PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})

	add_executable(benchmark_difference benchmark_difference.cpp)
	target_link_libraries(benchmark_difference ${GBENCHMARK_LIBRARIES})
	set_target_properties(benchmark_difference PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
else()
	message(STATUS "google benchmark not found, skipping benchmark programs")
endif()


# targets to analyze with IACA
if(IACA_FOUND)
	# intersection
	foreach(algo SCALAR SCALAR_COUNT BRANCHLESS BRANCHLESS_COUNT AVX2 AVX2_COUNT AVX512_CONFLICT AVX512)
		IACA_TARGET(IACA_INTERSECT_${algo} set_intersection.cpp)
	endforeach(algo)
	# difference
	foreach(algo AVX2 AVX512_CONFLICT)
		IACA_TARGET(IACA_DIFFERENCE_${algo} set_difference.cpp)
	endforeach(algo)
	# union
	foreach(algo BRANCHLESS)
		IACA_TARGET(IACA_UNION_${algo} set_union.cpp)
	endforeach(algo)
else()
	message(STATUS "iaca not found, disabled all iaca targets")
endif()
