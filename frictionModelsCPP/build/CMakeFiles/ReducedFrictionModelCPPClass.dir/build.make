# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/build

# Include any dependencies generated for this target.
include CMakeFiles/ReducedFrictionModelCPPClass.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ReducedFrictionModelCPPClass.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ReducedFrictionModelCPPClass.dir/flags.make

CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.o: CMakeFiles/ReducedFrictionModelCPPClass.dir/flags.make
CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.o: ../reducedModel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.o -c /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/reducedModel.cpp

CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/reducedModel.cpp > CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.i

CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/reducedModel.cpp -o CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.s

# Object files for target ReducedFrictionModelCPPClass
ReducedFrictionModelCPPClass_OBJECTS = \
"CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.o"

# External object files for target ReducedFrictionModelCPPClass
ReducedFrictionModelCPPClass_EXTERNAL_OBJECTS =

ReducedFrictionModelCPPClass.cpython-310-x86_64-linux-gnu.so: CMakeFiles/ReducedFrictionModelCPPClass.dir/reducedModel.cpp.o
ReducedFrictionModelCPPClass.cpython-310-x86_64-linux-gnu.so: CMakeFiles/ReducedFrictionModelCPPClass.dir/build.make
ReducedFrictionModelCPPClass.cpython-310-x86_64-linux-gnu.so: libmymodule_lib.so
ReducedFrictionModelCPPClass.cpython-310-x86_64-linux-gnu.so: libmymodule_lib2.so
ReducedFrictionModelCPPClass.cpython-310-x86_64-linux-gnu.so: CMakeFiles/ReducedFrictionModelCPPClass.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ReducedFrictionModelCPPClass.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ReducedFrictionModelCPPClass.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ReducedFrictionModelCPPClass.dir/build: ReducedFrictionModelCPPClass.cpython-310-x86_64-linux-gnu.so

.PHONY : CMakeFiles/ReducedFrictionModelCPPClass.dir/build

CMakeFiles/ReducedFrictionModelCPPClass.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ReducedFrictionModelCPPClass.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ReducedFrictionModelCPPClass.dir/clean

CMakeFiles/ReducedFrictionModelCPPClass.dir/depend:
	cd /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/build /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/build /home/gabriel/Documents/projects/FrictionModelling/frictionModelsCPP/build/CMakeFiles/ReducedFrictionModelCPPClass.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ReducedFrictionModelCPPClass.dir/depend
