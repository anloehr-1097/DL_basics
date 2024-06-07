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
  print_op(t1.op);
  t2.print();
  print_op(t2.op);
  t3.print();
  print_op(t3.op);
  Tensor<float> t4(1, 1, 4);
  exp_tensor(t1, &t4);
  t4.print();
  print_op(t4.op);
  Layer<float> layer_1(4,5,NonLinearity::SIGMOID);
  layer_1.calc(.4);
   layer_1.print();
  return 0;
};
