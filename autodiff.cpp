// Autodiff in C++ using operator overloading in the style of Karpathy's micrograd
#include<iostream>
#include<functional>
#include <tuple>
#include <cassert>
#include <vector>
#include "autodiff.h"

#define DEBUG 1

void print_op(Op op){
  switch (op.type) {
    case OpType::BINARY:
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
    };
      return;
  };
}


