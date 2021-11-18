echo "Building GraphIt G2 compiler"

cd graphit
mkdir -p build
cd build
cmake ..
make -j4

cd ../..

echo "Building BuilDSL"
cd BuilDSL
make -j8
cd ..
