cmake_minimum_required(VERSION 3.13)
#set(CMAKE_CXX_COMPILER "g++")
project(TCC LANGUAGES C CXX)
#project(TCC)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SOURCE_FILES
        main.cpp
        utils.cpp

        william.hpp
        william.cpp

        av1/ransac.h
        av1/ransac.c
        av1/mathutils.h
        av1/random.h

        av1/corner_match.c
        av1/corner_match.h
        av1/corner_detect.c
        av1/corner_detect.h

        av1/fastfeat/fast.c
        av1/fastfeat/fast.h
        av1/fastfeat/fast_9.c
        av1/fastfeat/nonmax.c

        opencv/cv.cpp
        opencv/draw.cpp
        opencv/draw.hpp

        robust/Base.cpp
        robust/AffineEstimator.cpp
        robust/RotZoomEstimator.cpp
        robust/TranslationEstimator.cpp
        robust/LMedS.cpp
        robust/PROSAC.cpp
        robust/MLESAC.cpp
        robust/MSAC.cpp
        robust/RANSAC.cpp
        robust/sac.hpp
        robust/sac.cpp

        robust/PROMSAC.cpp
        robust/PROMLESAC.cpp
        )

add_executable(main main.cpp ${SOURCE_FILES})

include_directories(/usr/local/bin)
target_link_libraries(main ${OpenCV_LIBS})
