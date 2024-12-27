#include "../ops.cpp"
#include <gtest/gtest.h>


TEST(OPS_TEST, test_attention){

    // init tensors
    Tensor<float> keys(1, 1, 4);
    Tensor<float> vals(1, 1, 8);
    Tensor<float> query(1, 1, 4);
    
    float k_ar[4] = {1.0, 2.0, 3.0, 4.0};
    float v_ar[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float q_ar[4] = {0.5, 0.25, 0, 0};

    keys.fill(k_ar, 4);
    query.fill(q_ar, 4);
    vals.fill(v_ar, 8);

    Tensor<float> res = attention(keys, vals, query);
    res.print();
    vals.print();


    EXPECT_TRUE(res==vals);
}

