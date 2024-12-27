#include "../autodiff.h"
#include <gtest/gtest.h>


TEST(TensorTest, comparison){
    Tensor<float> t1(1,2,3);
    Tensor<float> t2(1,2,3);
    EXPECT_NO_THROW(t1==t2);
}


TEST(TensorTest, copy_tensor){
    Tensor<float> t1(1, 2, 3);
    float t1_data[6] = {1,2,3,4,5,6};
    t1.fill(t1_data, 6);
    Tensor<float> t2 = t1;
    EXPECT_TRUE(t1==t2);
}

TEST(TensorTest, multiply){
    Tensor<float> t1(1,2,3);
    Tensor<float> t2(1,3,2);
    float t1_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float t2_data[6] = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    t1.fill(t1_data, 6);
    t2.fill(t2_data, 6);
    Tensor<float> res = t1 * t2;
    Tensor<float> exp_res(1,2,2);
    float data[4] = {4.0, 5.0, 10.0, 11.0};
    exp_res.fill(data, 4);
    EXPECT_TRUE(exp_res == res);
}
// 1 2 3
// 4 5 6
//
// 1 0
// 0 1
// 1 1


TEST(TensorTest, transpose){
    Tensor<int> t1(1,2,3);
    int t1_data[6] = {1, 2, 3, 4, 5, 6};
    t1.fill(t1_data, 6);
    Tensor<int> t3 = t1.transpose();
    Tensor<int> t2(1,3,2);
    int t2_data[6] = {1, 4, 2, 5, 3, 6};
    t2.fill(t2_data, 6);
    t3.print();
    t2.print();
    EXPECT_TRUE(t3 == t2);
}
