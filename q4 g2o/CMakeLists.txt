cmake_minimum_required(VERSION 3.0)
project(g2o_optimization)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(G2O REQUIRED)
include_directories( 
    ${G2O_INCLUDE_DIRS}
    "/usr/include/eigen3"
)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

# 添加可执行文件
add_executable(g2o_optimization g2o_optimization.cpp)

# 链接g2o库
target_link_libraries(g2o_optimization
    ${OpenCV_LIBS}
    g2o_core g2o_stuff)


