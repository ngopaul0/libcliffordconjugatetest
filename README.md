# Build instructions

## Prerequisites

On Ubuntu, you can install `libeigen3-dev`, but the build system will clone and build Eigen if you don't have it.

If using clang, install `libomp-dev` (for Ubuntu) for OpenMP support.

## Initializing

Initialize the CMake build directory:

```bash
$ cmake -DCMAKE_BUILD_TYPE=Release -B cmake-build-release
```

or for Debug builds

```bash
$ cmake -DCMAKE_BUILD_TYPE=Debug -B cmake-build-debug
```

You can just use any `-B` for the build directory, but directories named `cmake-build*` are covered by `.gitignore`.

## Running tests

The `-j` parameter specifies number of threads to use

```bash
$ cmake --build cmake-build-debug --target unittest -j 10
```

This will build an executable to run the tests in the newly created `cmake-build-debug` directory. Run tests by 

```bash
$ cmake-build-debug/unittest/unittest
```

Alternatively, do

```bash
$ cd cmake-build-debug
$ ctest
```

# Running benchmarks

Release builds are required for benchmarking for full optimization.

```bash
$ cmake --build cmake-build-release --target benchmark -j 10
$ cmake-build-release/benchmark/benchmark
```

You can run a specific suite of benchmarks and adjust benchmark samples to use per test, e.g.

```bash
$ cmake-build-release/benchmark/benchmark --benchmark-samples 1000 [algorithm]
$ cmake-build-release/benchmark/benchmark --benchmark-samples 100 [bruteforce]
```

Run `cmake-build-release/benchmark/benchmark -?` for a full list of options.
