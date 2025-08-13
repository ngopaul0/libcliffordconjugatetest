# Initializing

```bash
$ cmake -DCMAKE_BUILD_TYPE=Release -B cmake-build-release
```

or

```bash
$ cmake -DCMAKE_BUILD_TYPE=Debug -B cmake-build-debug
```

# Running tests

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

```bash
$ cmake --build cmake-build-release --target benchmark -j 10
$ cmake-build-debug/benchmark/benchmark
```
