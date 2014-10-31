# Copyright (c) 2014 Facebook.  All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file. See the AUTHORS file for names of contributors.

# Inherit some settings from environment variables, if available
INSTALL_PATH ?= $(CURDIR)

#-----------------------------------------------

ifneq ($(MAKECMDGOALS),dbg)
OPT += -O2 -fno-omit-frame-pointer -momit-leaf-frame-pointer
else
# intentionally left blank
endif

ifeq ($(MAKECMDGOALS),shared_lib)
OPT += -DNDEBUG
endif

ifeq ($(MAKECMDGOALS),static_lib)
OPT += -DNDEBUG
endif

#-----------------------------------------------

# detect what platform we're building on
$(shell (export ROCKETSPEED_ROOT="$(CURDIR)"; "$(CURDIR)/build_tools/build_detect_platform" "$(CURDIR)/build_config.mk"))
# this file is generated by the previous line to set build flags and sources
include build_config.mk

ifneq ($(PLATFORM), IOS)
CFLAGS += -g
CXXFLAGS += -g
else
# no debug info for IOS, that will make our library big
OPT += -DNDEBUG
endif

# ASAN doesn't work well with jemalloc. If we're compiling with ASAN, we should use regular malloc.
ifdef COMPILE_WITH_ASAN
	# ASAN compile flags
	EXEC_LDFLAGS += -fsanitize=address
	PLATFORM_CCFLAGS += -fsanitize=address
	PLATFORM_CXXFLAGS += -fsanitize=address
else
	# if we're not compiling with ASAN, use jemalloc
	EXEC_LDFLAGS := $(JEMALLOC_LIB) $(EXEC_LDFLAGS)
	PLATFORM_CXXFLAGS += $(JEMALLOC_INCLUDE) -DHAVE_JEMALLOC
	PLATFORM_CCFLAGS += $(JEMALLOC_INCLUDE) -DHAVE_JEMALLOC
endif

WARNING_FLAGS = -Wall -Werror -Wshadow
CFLAGS += $(WARNING_FLAGS) -I. -I./include $(PLATFORM_CCFLAGS) $(OPT)
CXXFLAGS += $(WARNING_FLAGS) -I. -I./include $(PLATFORM_CXXFLAGS) $(OPT) -Woverloaded-virtual

LDFLAGS += $(PLATFORM_LDFLAGS)

LIBOBJECTS = $(SOURCES:.cc=.o)
LIBOBJECTS += $(SOURCESCPP:.cpp=.o)

TESTUTIL = ./src/util/testutil.o
TESTCLUSTER = ./src/test/test_cluster.o
TESTHARNESS = ./src/util/testharness.o $(TESTUTIL) $(TESTCLUSTER)
TESTCONFIGURATION = ./src/test/configuration.o
BENCHHARNESS = ./src/util/benchharness.o
VALGRIND_ERROR = 2
VALGRIND_DIR = build_tools/VALGRIND_LOGS
VALGRIND_VER := $(join $(VALGRIND_VER),valgrind)
VALGRIND_OPTS = --error-exitcode=$(VALGRIND_ERROR) --leak-check=full

# constants for java
JC = javac
JARFLAGS = -cf
JAR = jar

TESTS = \
	autovector_test \
	arena_test \
	coding_test \
	env_test \
	consistent_hash_test \
	guid_generator_test \
	messages_test \
	auto_roll_logger_test \
  controlmessages_test \
  copilotmessages_test \
  pilotmessages_test \
  log_router_test \
  control_tower_router_test \
  mock_logdevice_test \
  logdevice_storage_test \
  hostmap_test \
  integration_test

TOOLS = \
	rocketbench

PROGRAMS = rocketspeed $(TOOLS)

# The library name is configurable since we are maintaining libraries of both
# debug/release mode.
ifeq ($(LIBNAME),)
        LIBNAME=librocketspeed
endif
LIBRARY = ${LIBNAME}.a

default: all

#-----------------------------------------------
# Create platform independent shared libraries.
#-----------------------------------------------
ifneq ($(PLATFORM_SHARED_EXT),)

ifneq ($(PLATFORM_SHARED_VERSIONED),true)
SHARED1 = ${LIBNAME}.$(PLATFORM_SHARED_EXT)
SHARED2 = $(SHARED1)
SHARED3 = $(SHARED1)
SHARED = $(SHARED1)
else
# Update RocketSpeed.h if you change these.
SHARED_MAJOR = 1
SHARED_MINOR = 1
SHARED1 = ${LIBNAME}.$(PLATFORM_SHARED_EXT)
SHARED2 = $(SHARED1).$(SHARED_MAJOR)
SHARED3 = $(SHARED1).$(SHARED_MAJOR).$(SHARED_MINOR)
SHARED = $(SHARED1) $(SHARED2) $(SHARED3)
$(SHARED1): $(SHARED3)
	ln -fs $(SHARED3) $(SHARED1)
$(SHARED2): $(SHARED3)
	ln -fs $(SHARED3) $(SHARED2)
endif

$(SHARED3):
	$(CXX) $(PLATFORM_SHARED_LDFLAGS)$(SHARED2) $(CXXFLAGS) $(PLATFORM_SHARED_CFLAGS) $(SOURCES) $(LDFLAGS) -o $@

endif  # PLATFORM_SHARED_EXT

.PHONY: blackbox_crash_test check clean coverage crash_test \
	release tags valgrind_check whitebox_crash_test format static_lib shared_lib all \
	dbg

all: $(LIBRARY) $(PROGRAMS) $(TESTS)

static_lib: $(LIBRARY)

shared_lib: $(SHARED)

dbg: $(LIBRARY) $(PROGRAMS) $(TESTS)

# creates static library and programs
release:
	$(MAKE) clean
	OPT="-DNDEBUG -O2" $(MAKE) all $(PROGRAMS) $(TOOLS) -j32

coverage:
	$(MAKE) clean
	COVERAGEFLAGS="-fprofile-arcs -ftest-coverage" LDFLAGS+="-lgcov" $(MAKE) all check -j32
	(cd coverage; ./coverage_test.sh)
	# Delete intermediate files
	find . -type f -regex ".*\.\(\(gcda\)\|\(gcno\)\)" -exec rm {} \;

# compile only the pilot
pilot: src/pilot/main.o $(LIBOBJECTS)
	$(CXX) src/pilot/main.o $(LIBOBJECTS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

# compile only the copilot
copilot: src/copilot/main.o $(LIBOBJECTS)
	$(CXX) src/copilot/main.o $(LIBOBJECTS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

# compile only the controltower
controltower: src/controltower/main.o $(LIBOBJECTS)
	$(CXX) src/controltower/main.o $(LIBOBJECTS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

# compile only the rocketspeed server
rocketspeed: src/server/main.o $(LIBOBJECTS)
	$(CXX) src/server/main.o $(LIBOBJECTS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

# compile only the rocketbench tool
rocketbench: src/tools/rocketbench/main.o $(LIBOBJECTS) $(TESTCONFIGURATION) $(TESTCLUSTER)
	$(CXX) src/tools/rocketbench/main.o $(LIBOBJECTS) $(TESTCONFIGURATION) $(TESTCLUSTER) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

# run all unit tests
check: $(TESTS)
	-rm -f test_times; \
	for t in $(TESTS); do \
		echo "***** Running $$t"; \
		./$$t || exit 1; \
	done; \
	echo ""; \
	echo "**** Slowest tests"; \
	cat test_times | sort -n -r | head -n10  # show 10 slowest tests

# test unexpected crashing of pilots, copilots and controltowers
crash_test:

asan_check:
	$(MAKE) clean
	COMPILE_WITH_ASAN=1 $(MAKE) check -j32
	$(MAKE) clean

asan_crash_test:
	$(MAKE) clean
	COMPILE_WITH_ASAN=1 $(MAKE) crash_test
	$(MAKE) clean

valgrind_check: all $(PROGRAMS) $(TESTS)
	mkdir -p $(VALGRIND_DIR)
	echo TESTS THAT HAVE VALGRIND ERRORS > $(VALGRIND_DIR)/valgrind_failed_tests; \
	echo TIMES in seconds TAKEN BY TESTS ON VALGRIND > $(VALGRIND_DIR)/valgrind_tests_times; \
	for t in $(filter-out skiplist_test,$(TESTS)); do \
		stime=`date '+%s'`; \
		$(VALGRIND_VER) $(VALGRIND_OPTS) ./$$t; \
		if [ $$? -eq $(VALGRIND_ERROR) ] ; then \
			echo $$t >> $(VALGRIND_DIR)/valgrind_failed_tests; \
		fi; \
		etime=`date '+%s'`; \
		echo $$t $$((etime - stime)) >> $(VALGRIND_DIR)/valgrind_tests_times; \
	done

java: rocketspeed.jar

rocketspeed.jar: RocketSpeedClient.class
	$(JAR) $(JARFLAGS) RocketSpeed.jar src/java/org/rocketspeed/RocketSpeedClient.class

RocketSpeedClient.class: src/java/org/rocketspeed/RocketSpeedClient.java
	$(JC) src/java/org/rocketspeed/RocketSpeedClient.java

clean:
	-rm -f $(PROGRAMS) $(TESTS) $(LIBRARY) $(SHARED) $(MEMENVLIBRARY) build_config.mk
	-rm -rf ios-x86/* ios-arm/*
	-rm -rf _mock_logdevice_logs
	-find src -name "*.[od]" -exec rm {} \;
	-find src -type f -regex ".*\.\(\(gcda\)\|\(gcno\)\)" -exec rm {} \;
	-find src/java -name "*.class" -exec rm {} \;
	-rm *.jar;

tags:
	ctags * -R
	cscope -b `find . -name '*.cc'` `find . -name '*.h'`

format:
	build_tools/format-diff.sh

coding_test: src/util/coding_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/coding_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

arena_test: src/util/arena_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/arena_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

autovector_test: src/util/autovector_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/autovector_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

env_test: src/util/env_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/env_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

consistent_hash_test: src/util/tests/consistent_hash_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/tests/consistent_hash_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

guid_generator_test: src/util/tests/guid_generator_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/tests/guid_generator_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

messages_test: src/messages/messages_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/messages/messages_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

auto_roll_logger_test: src/util/auto_roll_logger_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/auto_roll_logger_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

controlmessages_test: src/controltower/test/controlmessages_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/controltower/test/controlmessages_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

copilotmessages_test: src/copilot/test/copilotmessages_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/copilot/test/copilotmessages_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

pilotmessages_test: src/pilot/test/pilotmessages_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/pilot/test/pilotmessages_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

log_router_test: src/util/tests/log_router_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/tests/log_router_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

control_tower_router_test: src/util/tests/control_tower_router_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/tests/control_tower_router_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

mock_logdevice_test: src/logdevice/test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/logdevice/test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

logdevice_storage_test: src/util/logdevice_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/logdevice_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

hostmap_test: src/util/hostmap_test.o $(LIBOBJECTS) $(TESTHARNESS)
	$(CXX) src/util/hostmap_test.o $(LIBOBJECTS) $(TESTHARNESS) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

integration_test: src/test/integration_test.o $(LIBOBJECTS) $(TESTHARNESS) $(TESTCONFIGURATION)
	$(CXX) src/test/integration_test.o $(LIBOBJECTS) $(TESTHARNESS) $(TESTCONFIGURATION) $(EXEC_LDFLAGS) -o $@ $(LDFLAGS) $(COVERAGEFLAGS)

# ---------------------------------------------------------------------------
# 	Benchmarks and stress test
# ---------------------------------------------------------------------------
$(LIBRARY): $(LIBOBJECTS)
	rm -f $@
	$(AR) -rs $@ $(LIBOBJECTS)

rs_bench: db/rs_bench.o $(LIBOBJECTS) $(TESTUTIL)
	$(CXX) db/rs_bench.o $(LIBOBJECTS) $(TESTUTIL) $(EXEC_LDFLAGS) -o $@  $(LDFLAGS) $(COVERAGEFLAGS)

rs_stress: tools/rs_stress.o $(LIBOBJECTS) $(TESTUTIL)
	$(CXX) tools/rs_stress.o $(LIBOBJECTS) $(TESTUTIL) $(EXEC_LDFLAGS) -o $@  $(LDFLAGS) $(COVERAGEFLAGS)

# ---------------------------------------------------------------------------
#  	Platform-specific compilation
# ---------------------------------------------------------------------------

ifeq ($(PLATFORM), IOS)
# For iOS, create universal object files to be used on both the simulator and
# a device.
PLATFORMSROOT=/Applications/Xcode.app/Contents/Developer/Platforms
SIMULATORROOT=$(PLATFORMSROOT)/iPhoneSimulator.platform/Developer
DEVICEROOT=$(PLATFORMSROOT)/iPhoneOS.platform/Developer
IOSVERSION=$(shell defaults read $(PLATFORMSROOT)/iPhoneOS.platform/version CFBundleShortVersionString)

.cc.o:
	mkdir -p ios-x86/$(dir $@)
	$(CXX) $(CXXFLAGS) -isysroot $(SIMULATORROOT)/SDKs/iPhoneSimulator$(IOSVERSION).sdk -arch i686 -arch x86_64 -c $< -o ios-x86/$@
	mkdir -p ios-arm/$(dir $@)
	xcrun -sdk iphoneos $(CXX) $(CXXFLAGS) -isysroot $(DEVICEROOT)/SDKs/iPhoneOS$(IOSVERSION).sdk -arch armv6 -arch armv7 -arch armv7s -arch arm64 -c $< -o ios-arm/$@
	lipo ios-x86/$@ ios-arm/$@ -create -output $@

.c.o:
	mkdir -p ios-x86/$(dir $@)
	$(CC) $(CFLAGS) -isysroot $(SIMULATORROOT)/SDKs/iPhoneSimulator$(IOSVERSION).sdk -arch i686 -arch x86_64 -c $< -o ios-x86/$@
	mkdir -p ios-arm/$(dir $@)
	xcrun -sdk iphoneos $(CC) $(CFLAGS) -isysroot $(DEVICEROOT)/SDKs/iPhoneOS$(IOSVERSION).sdk -arch armv6 -arch armv7 -arch armv7s -arch arm64 -c $< -o ios-arm/$@
	lipo ios-x86/$@ ios-arm/$@ -create -output $@

else
.cc.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(COVERAGEFLAGS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@
endif

# ---------------------------------------------------------------------------
#  	Source files dependencies detection
# ---------------------------------------------------------------------------

# Add proper dependency support so changing a .h file forces a .cc file to
# rebuild.

# The .d file indicates .cc file's dependencies on .h files. We generate such
# dependency by g++'s -MM option, whose output is a make dependency rule.
# The sed command makes sure the "target" file in the generated .d file has
# the correct path prefix.
%.d: %.cc
	$(CXX) $(CXXFLAGS) $(PLATFORM_SHARED_CFLAGS) -MM $< -o $@
ifeq ($(PLATFORM), OS_MACOSX)
	@sed -i '' -e 's,.*:,$*.o:,' $@
else
	@sed -i -e 's,.*:,$*.o:,' $@
endif

DEPFILES = $(filter-out src/util/build_version.d,$(SOURCES:.cc=.d))

depend: $(DEPFILES)

# if the make goal is either "clean" or "format", we shouldn't
# try to import the *.d files.
# TODO(kailiu) The unfamiliarity of Make's conditions leads to the ugly
# working solution.
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),format)
ifneq ($(MAKECMDGOALS),jclean)
ifneq ($(MAKECMDGOALS),jtest)
-include $(DEPFILES)
endif
endif
endif
endif
