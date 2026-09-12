## cmake-test

This is just for manually testing/developing of a whisper.cpp installation to
enable troubleshooting issues and exploration. The idea is that this can be used
after making changes to whisper.cpp installation cmake configuration and then
verify it locally.

### Usage
The following will configure, build, and install whisper.cpp

Configuring/build/install:
```console
./build-install.sh
```
The above command will create a directory named `install` in the current directory
which will have the following files in its lib directory:
```console
$ ls install/lib/
cmake                   libggml-base.so         libggml.so                libparakeet.so          libwhisper.so
libggml-base.so.0       libggml-base.so.0.20.0  libggml.so.0              libparakeet.so.1        libwhisper.so.1
libggml-cpu.so          libggml.so.0.20.0       libggml.so.0.20.0         libparakeet.so.1.9.2    libwhisper.so.1.9.2
libggml-cpu.so.0        pkgconfig               libparakeet.so            libwhisper.so
```

Build/run this project using the installation created above:
```console
$ ./build.sh
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info - done
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features - done
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /path/to/whisper.cpp/examples/test-cmake/build
[ 50%] Building CXX object CMakeFiles/test-cmake.dir/test-cmake.cpp.o
[100%] Linking CXX executable test-cmake
[100%] Built target test-cmake
[test-cmake] version: 1.9.2-dev, build: 4925 (1fe009ca)
```
