cmake_minimum_required(VERSION 2.8.3)
project(rhoban_csa_mdp)

find_package(catkin REQUIRED COMPONENTS
  rhoban_fa
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rhoban_csa_mdp
  CATKIN_DEPENDS rhoban_fa
)

## Gaussian Processes support
# Currently, uses of GP in CSA_MDP is directly linked with the use of GP in
# regression forests. It would be possible to separate those uses if necessary
# in the future.
if ("${catkin_LIBRARIES}" MATCHES "rhoban_gp")
  add_definitions(-DRHOBAN_RF_USES_GP)
endif()


###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

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
add_library(rhoban_csa_mdp
  ${ALL_SOURCES}
  )

target_link_libraries(rhoban_csa_mdp
  ${catkin_LIBRARIES}
  )

set(TESTS
  kd_trees/kd_node
  kd_trees/kd_tree_container
  )

if (CATKIN_ENABLE_TESTING)
  # create an executable for each test
  foreach(test ${TESTS})
    string(REPLACE "/" "_" id ${test})
    catkin_add_gtest(${id} tests/${test}.cpp)
    target_link_libraries(${id} ${catkin_LIBRARIES} rhoban_csa_mdp)
  endforeach(test)
endif()
