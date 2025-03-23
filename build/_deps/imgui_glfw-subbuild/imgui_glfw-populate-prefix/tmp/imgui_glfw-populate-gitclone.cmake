# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

if(EXISTS "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-subbuild/imgui_glfw-populate-prefix/src/imgui_glfw-populate-stamp/imgui_glfw-populate-gitclone-lastrun.txt" AND EXISTS "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-subbuild/imgui_glfw-populate-prefix/src/imgui_glfw-populate-stamp/imgui_glfw-populate-gitinfo.txt" AND
  "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-subbuild/imgui_glfw-populate-prefix/src/imgui_glfw-populate-stamp/imgui_glfw-populate-gitclone-lastrun.txt" IS_NEWER_THAN "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-subbuild/imgui_glfw-populate-prefix/src/imgui_glfw-populate-stamp/imgui_glfw-populate-gitinfo.txt")
  message(VERBOSE
    "Avoiding repeated git clone, stamp file is up to date: "
    "'C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-subbuild/imgui_glfw-populate-prefix/src/imgui_glfw-populate-stamp/imgui_glfw-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

# Even at VERBOSE level, we don't want to see the commands executed, but
# enabling them to be shown for DEBUG may be useful to help diagnose problems.
cmake_language(GET_MESSAGE_LOG_LEVEL active_log_level)
if(active_log_level MATCHES "DEBUG|TRACE")
  set(maybe_show_command COMMAND_ECHO STDOUT)
else()
  set(maybe_show_command "")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-src"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: 'C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "C:/Program Files/Git/cmd/git.exe"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/ocornut/imgui.git" "imgui_glfw-src"
    WORKING_DIRECTORY "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps"
    RESULT_VARIABLE error_code
    ${maybe_show_command}
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(NOTICE "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/ocornut/imgui.git'")
endif()

execute_process(
  COMMAND "C:/Program Files/Git/cmd/git.exe"
          checkout "v1.89.9" --
  WORKING_DIRECTORY "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-src"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v1.89.9'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "C:/Program Files/Git/cmd/git.exe" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-src"
    RESULT_VARIABLE error_code
    ${maybe_show_command}
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: 'C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-subbuild/imgui_glfw-populate-prefix/src/imgui_glfw-populate-stamp/imgui_glfw-populate-gitinfo.txt" "C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-subbuild/imgui_glfw-populate-prefix/src/imgui_glfw-populate-stamp/imgui_glfw-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: 'C:/msys64/home/Comrade/robot_arm_ik_simulator/build/_deps/imgui_glfw-subbuild/imgui_glfw-populate-prefix/src/imgui_glfw-populate-stamp/imgui_glfw-populate-gitclone-lastrun.txt'")
endif()
