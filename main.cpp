#include <iostream>
//include "autodiff.cpp"
#include "autodiff.h"


int main(){
  Tensor<float> t1(1, 1, 4);
  Tensor<float> t2(1, 1, 4);
  Tensor<float> t3(1, 1, 4);
  float fill_data_1[4] = {1.0, 1.0, 1.0, 1.0};// {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  float fill_data_2[4] = {2.0, 2.0, 2.0, 2.0};// {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  t1.fill(fill_data_1, 4);
  t2.fill(fill_data_2, 4);
  add_tensors<float>(t1, t2, &t3);
  t1.print();
  t2.print();
  // print_op(t3.op);
  t3.print();
  return 0;
};
