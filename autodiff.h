// autodiff.h header file for autodiff.cpp



#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <iostream>
#include <cassert>
#include<functional>
#include <tuple>
#include <cassert>
#include <vector>


#define DEBUG 1


enum class UnaryOp {
	NEG,
	SIN,
	COS,
	EXP,
	LOG,
	TANH,
};
enum class BinaryOp {
  ADD,
  SUB,
  MUL,
  DIV,
  POW,
};

enum class NoneOp {
	NONE
};


union AnyOp {
	UnaryOp unary;
	BinaryOp binary;
  	NoneOp none;
};

enum class OpType {
	UNARY,
	BINARY,
	NONE

};

struct Op{
  	AnyOp op;
  	OpType type;
};

// create a tensor of type T
template<typename T>
class Tensor {
public:
  T *data = nullptr;
  int len = 0;
  std::tuple<int, int, int> shape;  // only 3D tensors for now
  std::vector<Tensor<T>*> preds;
  //Tensor<T> *preds = nullptr; // predecessors
  int n_preds = 0;
  Op op = {AnyOp{.none = NoneOp::NONE}, OpType::NONE};

  // Tensor class
  Tensor(int x, int y, int z){
    // assert valid dims
    assert(x > 0);
    assert(y > 0);
    assert(z > 0);

    shape = std::make_tuple(x, y, z);
    // heap allocate memory for tensor

    data = new T[x * y * z]{};
    len = x * y * z;


    if (DEBUG){
      std::cout << "Tensor created\n";
      std::cout << "Shape: " << std::get<0>(shape) << " " << std::get<1>(shape) << " " << std::get<2>(shape) << "\n";
      std::cout << "Length: " << len << "\n";
      for (int i = 0; i < len; i++){
	std::cout << data[i] << " ";
      };
      std::cout << "\n";
    };
	};

  ~Tensor(){
    delete[] data;
    //delete[] preds;
    if (DEBUG){
      std::cout << "Tensor deleted\n";
    };
  };

  // populate tensor with data
  void fill(T *fill_data, int fill_len){
    assert(fill_len == len);
    for (int i = 0; i < len; i++){
      this->data[i] = fill_data[i];
    };


  };

  // print tensor data
  void print(){
    std::cout << "Tensor Data of Tensor of len " << len << ":\n";
    for (int i = 0; i < len; i++){
      std::cout << data[i] << " ";
    };
    std::cout << std::endl;
  };
  // add preds to tensor binary operation
  void add_preds(Tensor<T> *t1, Tensor<T> *t2){
    preds.push_back(t1);
    preds.push_back(t2);
    n_preds += 2;
    op.type = OpType::BINARY;
    op.op.binary = BinaryOp::ADD;
  };
};


template<typename T>
void add_tensors(Tensor<T> &t1, Tensor<T> &t2, Tensor<T> *out){
  // add two tensors
  assert(t1.len == t2.len);
  for (int i = 0; i < t1.len; i++){
    out->data[i] = t1.data[i] + t2.data[i];
  };
  out->add_preds(&t1, &t2);

};


template<typename T>
void exp_tensor(Tensor<T> &t, Tensor<T> *out){
  // calculate exp of tensor entrywise
  for (int i = 0; i < t.len; i++){
    out -> data[i] = exp(t.data[i]);
  };
  out -> preds.push_back(&t);
  out -> n_preds = 1;
  out -> op.type = OpType::UNARY;
  out -> op.op.unary = UnaryOp::EXP;
};


void print_op(Op op);

#endif
