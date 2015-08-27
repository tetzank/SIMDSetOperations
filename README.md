# SIMDSetOperations
testbed for different SIMD implementations for set intersection and set union

This probably only works with gcc. Compile with:
```
$ mkdir build
$ cmake ..
$ make
```

This builds genlists, set_intersection and set_union. Before running any test
program you have to generate a test dataset with genlists. It will create
test.dat which is use by set_intersection and set_union. Parameters are in
constants.h.
