#include "autodiff.h"


int main(){
  Tensor<float> t1(1, 1, 4);
  Tensor<float> t2(1, 1, 4);
  Tensor<float> t3(1, 1, 4);
  float fill_data_1[4] = {1.0, 1.0, 1.0, 1.0};// {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  float fill_data_2[4] = {2.0, 2.0, 2.0, 2.0};// {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  float fill_data_3[5] = {1.0, 2.0, 3.0, 4.0, 5.0};// {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  double fill_data_4[5] = {1.0, 2.0, 3.0, 4.0, 5.0};// {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
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
  std::cout << "Create 1st Layer" << std::endl;
  Layer<float> layer_1(4, 5, NonLinearity::RELU);
  Tensor<float> t5(5,1,1);
  t5.fill(fill_data_3, 5);
  std::cout << "Compute 1st Layer" << std::endl;
  layer_1.compute(t5);
  layer_1.print();


  std::cout << "Create 2nd Layer" << std::endl;
  Tensor<double> t6(5,1,1);
  Layer<double> layer_2(4, 5, NonLinearity::RELU);
  t6.fill(fill_data_4, 5);

  std::cout << "Compute 2nd Layer" << std::endl;
  layer_2.compute(t6);
  layer_2.print();
  return 0;
};
