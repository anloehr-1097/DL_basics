// Autodiff in C++ using operator overloading in the style of Karpathy's micrograd
#include <algorithm>
#include<iostream>
#include <cassert>
#include "autodiff.h"

#ifndef DEBUG 
#define DEBUG 1
#endif

void print_op(Op op){
  switch (op.type) {
  case OpType::BINARY: {
      switch (op.op.binary) {
	case BinaryOp::ADD:
	    std::cout << "Add\n";
	    break;
	    return;

	case BinaryOp::SUB:
	    std::cout << "Sub\n";
	    break;
	    return;

	case BinaryOp::MUL:
	    std::cout << "Mul\n";
	    break;
	    return;

	case BinaryOp::DIV:
	    std::cout << "Div\n";
	    break;
	    return;

	case BinaryOp::POW:
	    std::cout << "Pow\n";
	    break;
	    return;
      };
      return;
  }

  case OpType::UNARY: {
	switch (op.op.unary) {
	case UnaryOp::NEG:
	  std::cout << "Neg\n";
	  break;
	  return;

	case UnaryOp::SIN:
	  std::cout << "Sin\n";
	  break;
	  return;

	case UnaryOp::COS:
	  std::cout << "Cos\n";
	  break;
	  return;

	case UnaryOp::EXP:
	  std::cout << "Exp\n";
	  break;
	  return;


	case UnaryOp::LOG:
	  std::cout << "Log\n";
	  break;
	  return;

	case UnaryOp::TANH:
	  std::cout << "Tanh\n";
	  break;
	  return;
	};
	return;
  }

    

  case OpType::NONE:
    std::cout << "None\n";
    break;
    return;

      return;
  };
}

template<typename T> 
T s_relu(T inp){
    return std::max(inp, (T)0);
}

Tensor<float> sigmoid(Tensor<float> &inp){
    std::cout << "SIGMOID" << std::endl;
    Tensor<float> out(std::get<0>(inp.shape),std::get<1>(inp.shape), std::get<2>(inp.shape));
    return out;
}

Tensor<float> f_sigmoid(Tensor<float> &inp){
    std::cout << "SIGMOID" << std::endl;
    Tensor<float> out(std::get<0>(inp.shape),std::get<1>(inp.shape), std::get<2>(inp.shape));
    return out;
}

Tensor<double> d_sigmoid(Tensor<double> &inp){
    std::cout << "SIGMOID" << std::endl;
    Tensor<double> out(std::get<0>(inp.shape),std::get<1>(inp.shape), std::get<2>(inp.shape));
    return out;
}


Tensor<double> d_relu(Tensor<double> &inp){

    Tensor<double> out(std::get<0>(inp.shape), std::get<1>(inp.shape), std::get<2>(inp.shape));
    for (int i = 0; i < inp.len; i++){
        out.data[i] = s_relu(inp.data[i]);
    }
    return out;
}

Tensor<float> relu(Tensor<float> &inp){

    Tensor<float> out(std::get<0>(inp.shape), std::get<1>(inp.shape), std::get<2>(inp.shape));
    for (int i = 0; i < inp.len; i++){
        out.data[i] = s_relu(inp.data[i]);
    }
    return out;
}


template<typename T>
Tensor<T> exponentiate(Tensor<T> &inp_tensor){
    Tensor<T> new_tens = inp_tensor;
    for (int i = 0; i < new_tens.len; i++) {
        new_tens.data[i] = exp(new_tens.data[i]);
    }

    return new_tens;
}


template<typename T>
Tensor<T> softmax(Tensor<T> &inp_tensor, int dim){
    // softmax function on tensor
    Tensor<T>intermed = exponentiate(inp_tensor);
    // TODO implement softmax row or columnwise
    return intermed / intermed.sum();
}



