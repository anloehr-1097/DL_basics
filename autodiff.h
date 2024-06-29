// autodiff.h header file for autodiff.cpp



#ifndef AUTODIFF_H
#define AUTODIFF_H
#endif

#include <cmath>
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


enum class NonLinearity {
    SIGMOID,
    RELU
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
  T *grad = nullptr; // gradient

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
    grad = new T[x * y * z]{};


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

   Tensor(){
    std::cout << "Empty init" << std::endl;

    };

  ~Tensor(){
     
    if (data == nullptr){
            return;
     }
    //delete[] preds;
    delete[] data;
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


// TODO implement these
Tensor<float> sigmoid(Tensor<float> &inp);
Tensor<float> f_sigmoid(Tensor<float> &inp);
Tensor<float> relu (Tensor<float> &inp);
Tensor<double> d_sigmoid(Tensor<double> &inp);
Tensor<double> d_relu (Tensor<double> &inp);


template<typename T>
Tensor<T> matmul(Tensor<T> &A, Tensor<T> &B){
  // basic non optimized matmul
  assert(std::get<1>(A.shape) == std::get<0>(B.shape));
  Tensor<T> C (std::get<0>(A.shape), std::get<1>(B.shape), 1);

  for (int i = 0; i < std::get<0>(A.shape); i++) {
    for (int j = 0; j < std::get<1>(B.shape); j++) {
      for (int k = 0; k < std::get<1>(A.shape); k++) {
	C.data[i * std::get<1>(C.shape) + j] += \
	  A.data[i * std::get<1>(A.shape) + k] * B.data[k * std::get<1>(B.shape) + j];
	  }
	}
  };
  return C;
}


// bundle inputs with ops
template<typename T>
class Layer {
    int num_in;
    int num_out;
    //using FuncPtr = T(*)(T);
    // FuncPtr act_fun_ptr = nullptr;
    Tensor<T> (*act_fun_ptr)(Tensor<T> &) = nullptr;
    Tensor<T> *data {};

public:
    Layer(int num_out_features, int num_in_features, NonLinearity act_fun){
        // create Matrix of rank, activation function
        //Tensor<T> data = Tensor<T>(num_in, num_out, 1);
        data = new Tensor<T>(num_out_features, num_in_features, 1);
        num_in = num_in_features;
        num_out = num_out_features;


        // check type T, assign correct function to act_fun_ptr
        if constexpr (std::is_same_v<T, float>){
            switch (act_fun){
                case NonLinearity::SIGMOID:
                    act_fun_ptr = &f_sigmoid;
                    break;
                case NonLinearity::RELU:
                    act_fun_ptr = &relu;
                    break;
                default:
                    act_fun_ptr = &f_sigmoid;
                    break;
            }
        }
        else if constexpr(std::is_same_v<T, double>) {
            // TODO double implementation of sigmoid
            switch (act_fun){
                case NonLinearity::SIGMOID:
                    act_fun_ptr = &d_sigmoid;
                    break;
                case NonLinearity::RELU:
                    act_fun_ptr = &d_relu;
                    break;
                default:
                    act_fun_ptr = &d_sigmoid;
                    break;
            }

        }
        else {
            std::cerr << "Unsupported type for Layer." << std::endl;
        }
    }

    Tensor<T> compute(Tensor<T> &inp){
        // assert that input is of expected size
        assert(std::get<0>(inp.shape) == std::get<1>(data->shape));  // matmul well defined
        assert(std::get<2>(inp.shape) == 1);  // 2d tensor

        //Tensor<T> out = matmul<T>(*data, inp);
        auto out = matmul<T>(*data, inp);
        return act_fun_ptr(out);
    };

    void print(){
        std::cout << "Data: ";
        data->print();
        std::cout << std::endl;
    };
};


// computation graph
template<typename T>
class Model {
public:
    int num_layers;
    Layer<T> *layers;
    Model(){
        ;

    };

    ~Model(){

    };
};

