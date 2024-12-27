#include "../ops.cpp"
#include <gtest/gtest.h>


TEST(OPS_TEST, test_name){
    Tensor<float> keys(1, 1, 4);
    float ar[4] = {1,2,3,4};
    keys.fill(ar, 4);
    keys.print();
    Tensor<float> vals(2, 2, 8);
    Tensor<float> query(1, 1, 4);
    Tensor<float> res = attention(keys, vals, query);
    Tensor<float> exp_res(1,1,4);
    EXPECT_TRUE(res==exp_res);
}

