# cmake_minimum_required(VERSION 3.5.1)
cmake_minimum_required(VERSION 3.10)

project(one_camera_track)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
add_definitions(-DSENSOR0_TYPE=OV_OS08A20_MIPI_8M_30FPS_12BIT)

set(CMAKE_C_COMPILER "aarch64-mix210-linux-gcc")
set(CMAKE_CXX_COMPILER "aarch64-mix210-linux-g++")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out")

add_compile_options(
  -std=c++11
  -fPIC
  -fstack-protector-all
  -D_FORTIFY_SOURCE=2 -O2
)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

set(SDK_PATH /home/liuyang/Documents/haisi/ss928_sdk_021)
set(MPP_PATH ${SDK_PATH}/smp/a55_linux/mpp)
set(SAMPLE_PATH ${MPP_PATH}/sample)
set(Ascend_DIR /usr/local/Ascend)
set(DDK_NNN_PATH ${Ascend_DIR}/ascend-toolkit/5.20)

# opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_part/opencv_kxw/lib/cmake/opencv4)

# FFmpeg路径
set(FFMPEG_PATH /home/liuyang/Documents/haisi/ffmpeg/install)

# eigen path
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/third_part/eigen-3.4.0)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

include_directories(
  ./include/
  ./include/svp/common/
  ${MPP_PATH}/out/include/
  ${SAMPLE_PATH}/common/
  ${DDK_NNN_PATH}/acllib/include/
  ${OpenCV_INCLUDE_DIRS}
  ${FFMPEG_PATH}/include
  ${SAMPLE_PATH}/audio/adp/

  ${EIGEN_PATH}/
)

set(LIB_NNN_PATH ${DDK_NNN_PATH}/acllib/lib64/stub/)
link_directories(${LIB_NNN_PATH} ${FFMPEG_PATH}/lib)
set(SYSTEM_LIBS pthread m dl stdc++)

add_library(hardware_decode STATIC src/ffmpeg_vdec_vpss.cpp)

FILE(GLOB OST_SRC ./src/nnn_ostrack_callback.cpp)
add_executable(test_ostrack ./examples/test_nnn_ostrack.cpp ${OST_SRC})

target_link_libraries(test_ostrack
  ${SYSTEM_LIBS}
  ascendcl
)

add_executable(test_ostrack_feature ./examples/test_nnn_ostrack_feature.cpp ${OST_SRC})
target_link_libraries(test_ostrack_feature
  ${SYSTEM_LIBS}
  ascendcl
)

add_executable(test_opencv ./examples/test_opencv.cpp)

target_link_libraries(test_opencv ${OpenCV_LIBS})


include(cmake/ive_link_libs.cmake)

FILE(GLOB SAMPLE_COMMON_SRC ${SAMPLE_PATH}/common/*.c)
FILE(GLOB SVP_COMMON_SRC ./src/svp/common/*.c)
FILE(GLOB UTILS_SRC ./src/utils.cpp)

add_executable(test_rtsp_vdec ./examples/test_rtsp_vdec.cpp ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${SVP_COMMON_SRC})
target_link_libraries(test_rtsp_vdec
  ${SYSTEM_LIBS}
  ${IVE_LINK_LIBS}
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)


FILE(GLOB TRACK_SRC ./src/Strack.cpp ./src/kalmanFilter.cpp)
add_executable(one_camera_track ./examples/one_camera_track.cpp ${SAMPLE_COMMON_SRC} ${UTILS_SRC} ${SVP_COMMON_SRC} ${OST_SRC} ${TRACK_SRC})
target_link_libraries(one_camera_track
  ${SYSTEM_LIBS}
  ${IVE_LINK_LIBS}
  ascendcl
  hardware_decode
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)
