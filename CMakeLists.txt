cmake_minimum_required(VERSION 2.8)
project(ActionRecognition)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
        src/Main.cpp
        src/FileManagement.cpp
        src/Performance.cpp
	src/FeatureExtractor.cpp
        src/action_utils.cpp
        src/Bboxes.cpp
        src/Model.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
