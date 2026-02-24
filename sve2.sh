mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCCMAKE_INSTALL_PREFIX=$(pwd) -DENABLE_SVE2=ON
make -j64