#include "../ops.cpp"
#include <gtest/gtest.h>

int main(){
    Tensor<float> keys(1, 1, 4);
    Tensor<float> vals(2, 2, 8);
    Tensor<float> query(1, 1, 4);
    Tensor<float> res;
    res = attention(keys, vals, query);

    res.print();
}
