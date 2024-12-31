#include "autodiff.cpp"
#include <cmath>
#include <iostream>


// define different operations on Tensors usually used in DL

template<typename T>
Tensor<T> attention(Tensor<T> &keys, Tensor<T> &values, Tensor<T> &queries){

    // keyy dim: 1xNxd_n
    // values dim: 1xNxd_m
    // queries dim: 1xMxd_n
    //
    if (std::get<2>(keys.shape) != std::get<2>(queries.shape)){
        std::cout << "Keys and values should be of same column size." << std::endl;
        throw -1;
    }

    float d_n = std::get<2>(keys.shape);
    Tensor<T> key_transp = keys.transpose();
    Tensor<T> scaled_dot_product = (queries * key_transp)/std::sqrt(d_n);
    scaled_dot_product.print();
    softmax(scaled_dot_product).print();

    Tensor<T> results = softmax(scaled_dot_product) * values;
    results.print();
    return results;
}
