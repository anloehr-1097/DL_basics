
//#include "../autodiff.cpp"
#include "../autodiff.h"

#include<iostream>

void op_test(){


  UnaryOp op_1 = UnaryOp::NEG;
  BinaryOp op_2 = BinaryOp::ADD;

  Op op = {};
  op.type = OpType::UNARY;
  op.op.unary = op_1;
  std::cout << "Printing op\n";
  print_op(op);

  op.type = OpType::BINARY;
  op.op.binary = op_2;
  std::cout << "Printing op\n";
  print_op(op);


}

int main(){
  op_test();
  return 0;

};


