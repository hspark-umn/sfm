# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/hspark/sfm_github/SfM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hspark/sfm_github/SfM/build

# Include any dependencies generated for this target.
include CMakeFiles/SfM.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SfM.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SfM.dir/flags.make

CMakeFiles/SfM.dir/SfM.cpp.o: CMakeFiles/SfM.dir/flags.make
CMakeFiles/SfM.dir/SfM.cpp.o: ../SfM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hspark/sfm_github/SfM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SfM.dir/SfM.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SfM.dir/SfM.cpp.o -c /home/hspark/sfm_github/SfM/SfM.cpp

CMakeFiles/SfM.dir/SfM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SfM.dir/SfM.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hspark/sfm_github/SfM/SfM.cpp > CMakeFiles/SfM.dir/SfM.cpp.i

CMakeFiles/SfM.dir/SfM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SfM.dir/SfM.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hspark/sfm_github/SfM/SfM.cpp -o CMakeFiles/SfM.dir/SfM.cpp.s

CMakeFiles/SfM.dir/SfM.cpp.o.requires:

.PHONY : CMakeFiles/SfM.dir/SfM.cpp.o.requires

CMakeFiles/SfM.dir/SfM.cpp.o.provides: CMakeFiles/SfM.dir/SfM.cpp.o.requires
	$(MAKE) -f CMakeFiles/SfM.dir/build.make CMakeFiles/SfM.dir/SfM.cpp.o.provides.build
.PHONY : CMakeFiles/SfM.dir/SfM.cpp.o.provides

CMakeFiles/SfM.dir/SfM.cpp.o.provides.build: CMakeFiles/SfM.dir/SfM.cpp.o


CMakeFiles/SfM.dir/CeresUtility.cpp.o: CMakeFiles/SfM.dir/flags.make
CMakeFiles/SfM.dir/CeresUtility.cpp.o: ../CeresUtility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hspark/sfm_github/SfM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/SfM.dir/CeresUtility.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SfM.dir/CeresUtility.cpp.o -c /home/hspark/sfm_github/SfM/CeresUtility.cpp

CMakeFiles/SfM.dir/CeresUtility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SfM.dir/CeresUtility.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hspark/sfm_github/SfM/CeresUtility.cpp > CMakeFiles/SfM.dir/CeresUtility.cpp.i

CMakeFiles/SfM.dir/CeresUtility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SfM.dir/CeresUtility.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hspark/sfm_github/SfM/CeresUtility.cpp -o CMakeFiles/SfM.dir/CeresUtility.cpp.s

CMakeFiles/SfM.dir/CeresUtility.cpp.o.requires:

.PHONY : CMakeFiles/SfM.dir/CeresUtility.cpp.o.requires

CMakeFiles/SfM.dir/CeresUtility.cpp.o.provides: CMakeFiles/SfM.dir/CeresUtility.cpp.o.requires
	$(MAKE) -f CMakeFiles/SfM.dir/build.make CMakeFiles/SfM.dir/CeresUtility.cpp.o.provides.build
.PHONY : CMakeFiles/SfM.dir/CeresUtility.cpp.o.provides

CMakeFiles/SfM.dir/CeresUtility.cpp.o.provides.build: CMakeFiles/SfM.dir/CeresUtility.cpp.o


CMakeFiles/SfM.dir/DataUtility.cpp.o: CMakeFiles/SfM.dir/flags.make
CMakeFiles/SfM.dir/DataUtility.cpp.o: ../DataUtility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hspark/sfm_github/SfM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/SfM.dir/DataUtility.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SfM.dir/DataUtility.cpp.o -c /home/hspark/sfm_github/SfM/DataUtility.cpp

CMakeFiles/SfM.dir/DataUtility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SfM.dir/DataUtility.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hspark/sfm_github/SfM/DataUtility.cpp > CMakeFiles/SfM.dir/DataUtility.cpp.i

CMakeFiles/SfM.dir/DataUtility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SfM.dir/DataUtility.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hspark/sfm_github/SfM/DataUtility.cpp -o CMakeFiles/SfM.dir/DataUtility.cpp.s

CMakeFiles/SfM.dir/DataUtility.cpp.o.requires:

.PHONY : CMakeFiles/SfM.dir/DataUtility.cpp.o.requires

CMakeFiles/SfM.dir/DataUtility.cpp.o.provides: CMakeFiles/SfM.dir/DataUtility.cpp.o.requires
	$(MAKE) -f CMakeFiles/SfM.dir/build.make CMakeFiles/SfM.dir/DataUtility.cpp.o.provides.build
.PHONY : CMakeFiles/SfM.dir/DataUtility.cpp.o.provides

CMakeFiles/SfM.dir/DataUtility.cpp.o.provides.build: CMakeFiles/SfM.dir/DataUtility.cpp.o


CMakeFiles/SfM.dir/epnp.cpp.o: CMakeFiles/SfM.dir/flags.make
CMakeFiles/SfM.dir/epnp.cpp.o: ../epnp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hspark/sfm_github/SfM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/SfM.dir/epnp.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SfM.dir/epnp.cpp.o -c /home/hspark/sfm_github/SfM/epnp.cpp

CMakeFiles/SfM.dir/epnp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SfM.dir/epnp.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hspark/sfm_github/SfM/epnp.cpp > CMakeFiles/SfM.dir/epnp.cpp.i

CMakeFiles/SfM.dir/epnp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SfM.dir/epnp.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hspark/sfm_github/SfM/epnp.cpp -o CMakeFiles/SfM.dir/epnp.cpp.s

CMakeFiles/SfM.dir/epnp.cpp.o.requires:

.PHONY : CMakeFiles/SfM.dir/epnp.cpp.o.requires

CMakeFiles/SfM.dir/epnp.cpp.o.provides: CMakeFiles/SfM.dir/epnp.cpp.o.requires
	$(MAKE) -f CMakeFiles/SfM.dir/build.make CMakeFiles/SfM.dir/epnp.cpp.o.provides.build
.PHONY : CMakeFiles/SfM.dir/epnp.cpp.o.provides

CMakeFiles/SfM.dir/epnp.cpp.o.provides.build: CMakeFiles/SfM.dir/epnp.cpp.o


CMakeFiles/SfM.dir/MathUtility.cpp.o: CMakeFiles/SfM.dir/flags.make
CMakeFiles/SfM.dir/MathUtility.cpp.o: ../MathUtility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hspark/sfm_github/SfM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/SfM.dir/MathUtility.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SfM.dir/MathUtility.cpp.o -c /home/hspark/sfm_github/SfM/MathUtility.cpp

CMakeFiles/SfM.dir/MathUtility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SfM.dir/MathUtility.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hspark/sfm_github/SfM/MathUtility.cpp > CMakeFiles/SfM.dir/MathUtility.cpp.i

CMakeFiles/SfM.dir/MathUtility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SfM.dir/MathUtility.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hspark/sfm_github/SfM/MathUtility.cpp -o CMakeFiles/SfM.dir/MathUtility.cpp.s

CMakeFiles/SfM.dir/MathUtility.cpp.o.requires:

.PHONY : CMakeFiles/SfM.dir/MathUtility.cpp.o.requires

CMakeFiles/SfM.dir/MathUtility.cpp.o.provides: CMakeFiles/SfM.dir/MathUtility.cpp.o.requires
	$(MAKE) -f CMakeFiles/SfM.dir/build.make CMakeFiles/SfM.dir/MathUtility.cpp.o.provides.build
.PHONY : CMakeFiles/SfM.dir/MathUtility.cpp.o.provides

CMakeFiles/SfM.dir/MathUtility.cpp.o.provides.build: CMakeFiles/SfM.dir/MathUtility.cpp.o


# Object files for target SfM
SfM_OBJECTS = \
"CMakeFiles/SfM.dir/SfM.cpp.o" \
"CMakeFiles/SfM.dir/CeresUtility.cpp.o" \
"CMakeFiles/SfM.dir/DataUtility.cpp.o" \
"CMakeFiles/SfM.dir/epnp.cpp.o" \
"CMakeFiles/SfM.dir/MathUtility.cpp.o"

# External object files for target SfM
SfM_EXTERNAL_OBJECTS =

SfM: CMakeFiles/SfM.dir/SfM.cpp.o
SfM: CMakeFiles/SfM.dir/CeresUtility.cpp.o
SfM: CMakeFiles/SfM.dir/DataUtility.cpp.o
SfM: CMakeFiles/SfM.dir/epnp.cpp.o
SfM: CMakeFiles/SfM.dir/MathUtility.cpp.o
SfM: CMakeFiles/SfM.dir/build.make
SfM: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
SfM: /usr/local/lib/libceres.a
SfM: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
SfM: /usr/lib/x86_64-linux-gnu/libglog.so
SfM: /usr/lib/x86_64-linux-gnu/libgflags.so
SfM: /usr/lib/x86_64-linux-gnu/libspqr.so
SfM: /usr/lib/x86_64-linux-gnu/libtbb.so
SfM: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
SfM: /usr/lib/x86_64-linux-gnu/libcholmod.so
SfM: /usr/lib/x86_64-linux-gnu/libccolamd.so
SfM: /usr/lib/x86_64-linux-gnu/libcamd.so
SfM: /usr/lib/x86_64-linux-gnu/libcolamd.so
SfM: /usr/lib/x86_64-linux-gnu/libamd.so
SfM: /usr/lib/liblapack.so
SfM: /usr/lib/libf77blas.so
SfM: /usr/lib/libatlas.so
SfM: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
SfM: /usr/lib/x86_64-linux-gnu/librt.so
SfM: /usr/lib/x86_64-linux-gnu/libcxsparse.so
SfM: /usr/lib/x86_64-linux-gnu/libspqr.so
SfM: /usr/lib/x86_64-linux-gnu/libtbb.so
SfM: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
SfM: /usr/lib/x86_64-linux-gnu/libcholmod.so
SfM: /usr/lib/x86_64-linux-gnu/libccolamd.so
SfM: /usr/lib/x86_64-linux-gnu/libcamd.so
SfM: /usr/lib/x86_64-linux-gnu/libcolamd.so
SfM: /usr/lib/x86_64-linux-gnu/libamd.so
SfM: /usr/lib/liblapack.so
SfM: /usr/lib/libf77blas.so
SfM: /usr/lib/libatlas.so
SfM: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
SfM: /usr/lib/x86_64-linux-gnu/librt.so
SfM: /usr/lib/x86_64-linux-gnu/libcxsparse.so
SfM: CMakeFiles/SfM.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hspark/sfm_github/SfM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable SfM"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SfM.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SfM.dir/build: SfM

.PHONY : CMakeFiles/SfM.dir/build

CMakeFiles/SfM.dir/requires: CMakeFiles/SfM.dir/SfM.cpp.o.requires
CMakeFiles/SfM.dir/requires: CMakeFiles/SfM.dir/CeresUtility.cpp.o.requires
CMakeFiles/SfM.dir/requires: CMakeFiles/SfM.dir/DataUtility.cpp.o.requires
CMakeFiles/SfM.dir/requires: CMakeFiles/SfM.dir/epnp.cpp.o.requires
CMakeFiles/SfM.dir/requires: CMakeFiles/SfM.dir/MathUtility.cpp.o.requires

.PHONY : CMakeFiles/SfM.dir/requires

CMakeFiles/SfM.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SfM.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SfM.dir/clean

CMakeFiles/SfM.dir/depend:
	cd /home/hspark/sfm_github/SfM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hspark/sfm_github/SfM /home/hspark/sfm_github/SfM /home/hspark/sfm_github/SfM/build /home/hspark/sfm_github/SfM/build /home/hspark/sfm_github/SfM/build/CMakeFiles/SfM.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SfM.dir/depend
