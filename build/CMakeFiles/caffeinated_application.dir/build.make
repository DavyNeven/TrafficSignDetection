# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

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
CMAKE_SOURCE_DIR = /users/visics/dneven/ClionProjects/testCaffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /users/visics/dneven/ClionProjects/testCaffe/build

# Include any dependencies generated for this target.
include CMakeFiles/caffeinated_application.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/caffeinated_application.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/caffeinated_application.dir/flags.make

CMakeFiles/caffeinated_application.dir/main.cpp.o: CMakeFiles/caffeinated_application.dir/flags.make
CMakeFiles/caffeinated_application.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/visics/dneven/ClionProjects/testCaffe/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/caffeinated_application.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/caffeinated_application.dir/main.cpp.o -c /users/visics/dneven/ClionProjects/testCaffe/main.cpp

CMakeFiles/caffeinated_application.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffeinated_application.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/visics/dneven/ClionProjects/testCaffe/main.cpp > CMakeFiles/caffeinated_application.dir/main.cpp.i

CMakeFiles/caffeinated_application.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffeinated_application.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/visics/dneven/ClionProjects/testCaffe/main.cpp -o CMakeFiles/caffeinated_application.dir/main.cpp.s

CMakeFiles/caffeinated_application.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/caffeinated_application.dir/main.cpp.o.requires

CMakeFiles/caffeinated_application.dir/main.cpp.o.provides: CMakeFiles/caffeinated_application.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/caffeinated_application.dir/build.make CMakeFiles/caffeinated_application.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/caffeinated_application.dir/main.cpp.o.provides

CMakeFiles/caffeinated_application.dir/main.cpp.o.provides.build: CMakeFiles/caffeinated_application.dir/main.cpp.o

CMakeFiles/caffeinated_application.dir/Classifier.cpp.o: CMakeFiles/caffeinated_application.dir/flags.make
CMakeFiles/caffeinated_application.dir/Classifier.cpp.o: ../Classifier.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /users/visics/dneven/ClionProjects/testCaffe/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/caffeinated_application.dir/Classifier.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/caffeinated_application.dir/Classifier.cpp.o -c /users/visics/dneven/ClionProjects/testCaffe/Classifier.cpp

CMakeFiles/caffeinated_application.dir/Classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffeinated_application.dir/Classifier.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /users/visics/dneven/ClionProjects/testCaffe/Classifier.cpp > CMakeFiles/caffeinated_application.dir/Classifier.cpp.i

CMakeFiles/caffeinated_application.dir/Classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffeinated_application.dir/Classifier.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /users/visics/dneven/ClionProjects/testCaffe/Classifier.cpp -o CMakeFiles/caffeinated_application.dir/Classifier.cpp.s

CMakeFiles/caffeinated_application.dir/Classifier.cpp.o.requires:
.PHONY : CMakeFiles/caffeinated_application.dir/Classifier.cpp.o.requires

CMakeFiles/caffeinated_application.dir/Classifier.cpp.o.provides: CMakeFiles/caffeinated_application.dir/Classifier.cpp.o.requires
	$(MAKE) -f CMakeFiles/caffeinated_application.dir/build.make CMakeFiles/caffeinated_application.dir/Classifier.cpp.o.provides.build
.PHONY : CMakeFiles/caffeinated_application.dir/Classifier.cpp.o.provides

CMakeFiles/caffeinated_application.dir/Classifier.cpp.o.provides.build: CMakeFiles/caffeinated_application.dir/Classifier.cpp.o

# Object files for target caffeinated_application
caffeinated_application_OBJECTS = \
"CMakeFiles/caffeinated_application.dir/main.cpp.o" \
"CMakeFiles/caffeinated_application.dir/Classifier.cpp.o"

# External object files for target caffeinated_application
caffeinated_application_EXTERNAL_OBJECTS =

caffeinated_application: CMakeFiles/caffeinated_application.dir/main.cpp.o
caffeinated_application: CMakeFiles/caffeinated_application.dir/Classifier.cpp.o
caffeinated_application: CMakeFiles/caffeinated_application.dir/build.make
caffeinated_application: /users/visics/dneven/caffe-master-tbd/caffe-master/cmake_build/lib/libcaffe.so
caffeinated_application: /users/visics/dneven/caffe-master-tbd/caffe-master/cmake_build/lib/libproto.a
caffeinated_application: /usr/lib64/libboost_system.so
caffeinated_application: /usr/lib64/libboost_thread.so
caffeinated_application: /usr/lib64/libglog.so
caffeinated_application: /usr/lib64/libgflags.so
caffeinated_application: /usr/lib64/libprotobuf.so
caffeinated_application: /usr/lib64/libglog.so
caffeinated_application: /usr/lib64/libgflags.so
caffeinated_application: /usr/lib64/libprotobuf.so
caffeinated_application: /usr/lib64/libhdf5_hl.so
caffeinated_application: /usr/lib64/libhdf5.so
caffeinated_application: /usr/lib64/liblmdb.so
caffeinated_application: /usr/lib64/libleveldb.so
caffeinated_application: /usr/lib64/libsnappy.so
caffeinated_application: /usr/local/cuda-7.5/lib64/libcudart.so
caffeinated_application: /usr/local/cuda-7.5/lib64/libcurand.so
caffeinated_application: /usr/local/cuda-7.5/lib64/libcublas.so
caffeinated_application: /usr/lib64/libopencv_highgui.so.2.4.9
caffeinated_application: /usr/lib64/libopencv_imgproc.so.2.4.9
caffeinated_application: /usr/lib64/libopencv_core.so.2.4.9
caffeinated_application: /esat/matar/ttommasi/intel/mkl/lib/intel64/libmkl_rt.so
caffeinated_application: /usr/lib64/libpython2.7.so
caffeinated_application: /usr/lib64/libboost_python.so
caffeinated_application: CMakeFiles/caffeinated_application.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable caffeinated_application"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffeinated_application.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/caffeinated_application.dir/build: caffeinated_application
.PHONY : CMakeFiles/caffeinated_application.dir/build

CMakeFiles/caffeinated_application.dir/requires: CMakeFiles/caffeinated_application.dir/main.cpp.o.requires
CMakeFiles/caffeinated_application.dir/requires: CMakeFiles/caffeinated_application.dir/Classifier.cpp.o.requires
.PHONY : CMakeFiles/caffeinated_application.dir/requires

CMakeFiles/caffeinated_application.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/caffeinated_application.dir/cmake_clean.cmake
.PHONY : CMakeFiles/caffeinated_application.dir/clean

CMakeFiles/caffeinated_application.dir/depend:
	cd /users/visics/dneven/ClionProjects/testCaffe/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /users/visics/dneven/ClionProjects/testCaffe /users/visics/dneven/ClionProjects/testCaffe /users/visics/dneven/ClionProjects/testCaffe/build /users/visics/dneven/ClionProjects/testCaffe/build /users/visics/dneven/ClionProjects/testCaffe/build/CMakeFiles/caffeinated_application.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/caffeinated_application.dir/depend

