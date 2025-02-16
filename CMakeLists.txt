cmake_minimum_required(VERSION 3.10)
project(RobotArm)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

# Auto-download GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
FetchContent_MakeAvailable(glfw)

# Auto-download GLAD (OpenGL loader)
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v0.1.36
)
FetchContent_MakeAvailable(glad)

# Auto-download GLM (Linear Algebra)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.8
)
FetchContent_MakeAvailable(glm)

# Auto-download ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.89.9  # Change to the latest stable release if needed
)
FetchContent_MakeAvailable(imgui)

# Auto-download ImGui GLFW backend
FetchContent_Declare(
    imgui_glfw
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.89.9
    SOURCE_SUBDIR backends
)
FetchContent_MakeAvailable(imgui_glfw)

# Find OpenGL (System Dependency)
find_package(OpenGL REQUIRED)

# Add Source Files
add_executable(RobotArm src/main.cpp)

# Include Fetched Dependencies
include_directories(${glfw_SOURCE_DIR}/include ${glm_SOURCE_DIR} ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)

# Link Libraries
add_library(imgui_lib 
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# Link Libraries to Executable
target_link_libraries(RobotArm OpenGL::GL glfw glad imgui_lib)