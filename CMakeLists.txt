cmake_minimum_required(VERSION 3.16.3)
project(rhoban_csa_mdp)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17")

## Gaussian Processes support
# Currently, uses of GP in CSA_MDP is directly linked with the use of GP in
# regression forests. It would be possible to separate those uses if necessary
# in the future.
#if ("${catkin_LIBRARIES}" MATCHES "rhoban_gp")
#  add_definitions(-DRHOBAN_RF_USES_GP)
#endif()
# ↑↑↑↑↑↑↑ NOW PROPAGATE AUTOMATICALY BY CMAKE

# Directories containing sources files
set(DIRECTORIES
  src/rhoban_csa_mdp/action_optimizers
  src/rhoban_csa_mdp/core
  src/rhoban_csa_mdp/knownness
  src/rhoban_csa_mdp/online_planners
  src/rhoban_csa_mdp/reward_predictors
  src/rhoban_csa_mdp/value_approximators
  src/rhoban_csa_mdp/solvers
  src/kd_trees# Should be elsewhere...
  )

# Build ALL_SOURCES
set (SOURCES)
set (ALL_SOURCES)
set (PREFIXED_SOURCES)
foreach (DIRECTORY ${DIRECTORIES})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}")
    include (${DIRECTORY}/sources.cmake)

    foreach (SOURCE ${SOURCES})
        set (PREFIXED_SOURCES ${PREFIXED_SOURCES} ${DIRECTORY}/${SOURCE})
    endforeach (SOURCE)

    set (ALL_SOURCES ${ALL_SOURCES} ${PREFIXED_SOURCES})
endforeach (DIRECTORY)

# Declare a C++ library
add_library(rhoban_csa_mdp ${ALL_SOURCES})
target_link_libraries(rhoban_csa_mdp PUBLIC rhoban_fa)
target_include_directories(rhoban_csa_mdp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# optional
if(TARGET rhoban_gp)
  target_link_libraries(rhoban_csa_mdp PUBLIC rhoban_gp)
endif()

# optional 
if(TARGET rhoban_viewer)
  target_link_libraries(rhoban_csa_mdp PUBLIC rhoban_viewer)
endif()


