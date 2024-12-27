#include "autodiff.h"
#include <iostream>


// define different operations on Tensors usually used in DL

template<typename T>
Tensor<T> attention(Tensor<T> &keys, Tensor<T> &values, Tensor<T> &query){
    std::cout << "attention";
    keys.print();
    std::cout << "attention END";
    return keys;

}
