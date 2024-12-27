#include "autodiff.h"
#include <iostream>


// define different operations on Tensors usually used in DL

template<typename T>
Tensor<T> attention(Tensor<T> &keys, Tensor<T> &values, Tensor<T> &queries){

    // keyy dim: Nxn
    // values dim: Nxm
    // queries dim: Mxn
    //
    if (std::get<2>(keys.shape) != std::get<2>(queries.shape)){
        std::cout << "Keys and values should be of same column size." << std::endl;
        throw -1;
    }

    Tensor<T> key_transp = keys.transpose();
    keys.print();
    key_transp.print();
    values.print();
    queries.print();
    Tensor<T> scaled_dot_product = queries * key_transp; 
    Tensor<T> results = scaled_dot_product * values;

    return scaled_dot_product * values;
    

}
