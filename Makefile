# This makefile will build the C++ components of Deepbinner.

# Example commands:
#   make (build in release mode)
#   make debug (build in debug mode)
#   make clean (deletes *.o files, which aren't required to run the aligner)
#   make distclean (deletes *.o files and the *.so file, which is required to run the aligner)
#   make CXX=g++-5 (build with a particular compiler)
#   make CXXFLAGS="-Werror -g3" (build with particular compiler flags)


# Determine the platform
ifeq ($(shell uname), Darwin)
  PLATFORM = Mac
else
  PLATFORM = Linux
endif
$(info Platform: $(PLATFORM))


# Determine the compiler and version
COMPILER_HELP := $(shell $(CXX) --help | head -n 1)
ifneq (,$(findstring clang,$(COMPILER_HELP)))
    COMPILER = clang
else ifneq (,$(findstring g++,$(COMPILER_HELP)))
    COMPILER = g++
else ifneq (,$(findstring Intel,$(COMPILER_HELP)))
    COMPILER = icpc
else
    COMPILER = unknown
endif
ifeq ($(COMPILER),clang)
  COMPILER_VERSION := $(shell $(CXX) --version | grep version | grep -o -m 1 "[0-9]\+\.[0-9]\+\.*[0-9]*" | head -n 1)
else
  COMPILER_VERSION := $(shell $(CXX) -dumpfullversion 2> /dev/null)
  ifeq ($(COMPILER_VERSION),)
      $(info Falling back to -dumpversion as compiler did not support -dumpfullversion)
      COMPILER_VERSION := $(shell $(CXX) -dumpversion)
  endif
endif
$(info Compiler: $(COMPILER) $(COMPILER_VERSION))


# CXXFLAGS can be overridden by the user.
CXXFLAGS    ?= -Wall -Wextra -pedantic -mtune=native


# These flags are required for the build to work.
FLAGS        = -std=c++11 -fPIC
LDFLAGS      = -shared -lz


# Different debug/optimisation levels for debug/release builds.
DEBUGFLAGS   = -g
RELEASEFLAGS = -O3 -DNDEBUG


TARGET       = deepbinner/dtw/dtw.so
SHELL        = /bin/sh
SOURCES      = $(shell find deepbinner/dtw -name "*.cpp")
HEADERS      = $(shell find deepbinner/dtw -name "*.h")
OBJECTS      = $(SOURCES:.cpp=.o)


# Linux needs '-soname' while Mac needs '-install_name'
ifeq ($(PLATFORM), Mac)
  SONAME       = -install_name
else  # Linux
  SONAME       = -soname
endif


.PHONY: release
release: FLAGS+=$(RELEASEFLAGS)
release: $(TARGET)


.PHONY: debug
debug: FLAGS+=$(DEBUGFLAGS)
debug: $(TARGET)


$(TARGET): $(OBJECTS)
	$(CXX) $(FLAGS) $(CXXFLAGS) -Wl,$(SONAME),$(TARGET) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

clean:
	$(RM) $(OBJECTS)

distclean: clean
	$(RM) $(TARGET)

%.o: %.cpp $(HEADERS)
	$(CXX) $(FLAGS) $(CXXFLAGS) -c -o $@ $<
