c++ -I BuilDSL/buildit/include -I BuilDSL/include mm-dsl.cpp -o dsl.out -L BuilDSL/buildit/build  -L BuilDSL/build -lgraphit -lbuildit -rdynamic || exit
./dsl.out
