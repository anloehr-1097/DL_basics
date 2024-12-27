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
    t1.print();
    Tensor<float> t2 = t1;
    EXPECT_TRUE(t1==t2);
}
