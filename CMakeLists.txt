cmake_minimum_required(VERSION 3.5.1)

project(one_camera_track)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

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
set(Ascend_DIR /usr/local/Ascend)
set(DDK_NNN_PATH ${Ascend_DIR}/ascend-toolkit/5.20)

include_directories(
  ./include/
  ${DDK_NNN_PATH}/acllib/include/
)

set(LIB_NNN_PATH ${DDK_NNN_PATH}/acllib/lib64/stub/)
link_directories(${LIB_NNN_PATH})
set(SYSTEM_LIBS pthread m dl stdc++)

FILE(GLOB src_file ./src/*.cpp)

add_executable(test_ostrack ./examples/test_nnn_ostrack.cpp ${src_file})

target_link_libraries(test_ostrack
  ${SYSTEM_LIBS}
  ascendcl
)
