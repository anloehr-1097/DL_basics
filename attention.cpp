#include <cstddef>
#include <iostream>
#include <ostream>
#include <vector>
#include <cassert> 
#include <cmath>
#include <numeric>
#include<random>


template<typename T>
class Matrix {

public:
  size_t rows = 0;
  size_t cols = 0;
  std::vector<T> data = {};

  Matrix(size_t r, size_t c){
    rows = r;
    cols = c;
    data = std::vector<T>(rows*cols);
  };

  void populate(std::vector<T> vec){
    data = vec;
  };

  void transpose(){
    // transpose the matrix
    std::vector<T> new_data = std::vector<T>(rows*cols);
    for (int i = 0; i < rows*cols; i++) {
      int new_col = i / cols;
      int new_row = i - (i / cols) * cols;
      
      new_data[new_row * cols + new_col] = data[i];
    };
    data = new_data;
  };


  void operator/(T num){
    assert(num != 0);
    for (int i = 0; i < data.size(); i++) {
      data[i] /= num;
	}
  };

  void print(){
    std::cout << "Matrix Data\n";
    for (int i = 0; i < data.size(); i++){
      std::cout << data[i] << " ";
    };
    std::cout << std::endl;

  };
};

template<typename T>
Matrix<T> matmul(Matrix<T> A, Matrix<T> B){
  // basic non optimized matmul
  assert(A.cols == B.rows);

  Matrix<T> C = Matrix<T>(A.rows, B.cols);

  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < B.cols; j++) {
      for (int k = 0; k < A.cols; k++) {
	C.data[i * C.cols + j] += \
	  A.data[i * A.cols + k] * B.data[k * B.cols + j];
	  }
	}
  };
  return C;
}


template<typename T> T dot(std::vector<T> v1, std::vector<T> v2){
  // only for numeric types
  T res = {};
  for (int i = 0; i < v1.size(); ++i){
    res += v1[i] * v2[i];
  }

};


template<typename T>
std::vector<T> softmax(std::vector<T> v){
  for (int i = 0; i < v.size(); i++) {
    v[i] = std::exp(v[i]);
  };
  T v_sum  = std::accumulate(v.begin(), v.end(), 0);
  for (int i = 0; i < v.size(); i++) {
    v[i] /= v_sum;
  };
  return v;
  
};

template<typename T>
void random_populate(std::vector<T> *v, std::uniform_real_distribution<T> &d, std::default_random_engine dev){
  for (int i = 0; i < v->size(); i++){
    v->at(i) = d(dev);};

};

template<typename T>
Matrix<T> softmax(Matrix<T> m){
  // apply softmax rowwise to matrix
  Matrix<T> rw_sm_mat = Matrix<T>(m.rows, m.cols);
  std::vector<T> new_data = {};
  std::vector<T> v(m.cols);

  typename std::vector<T>::iterator it = m.data.begin();

  for (int i = 0; i < m.rows; i++){
    for (int j = 0; j < m.cols; j++){
      v[j] = m.data[i * m.cols + j];
    };
    v = softmax(v);
    new_data.insert(new_data.end(), v.begin(), v.end());

  };
  rw_sm_mat.populate(new_data);
  rw_sm_mat.print();
  return rw_sm_mat;
};



template <typename T>
std::vector<T> attention(Matrix<T> keys, Matrix<T> values, Matrix<T> queries){
  // queries and keys rowwise in matrix
  keys.transpose();
  Matrix<T> index_mat = matmul(queries, keys);
  index_mat.print();
  index_mat / T(keys.cols);
  index_mat.print();
  // TODO apply rowwise softmax
  std::cout << "Softmax\n";
  index_mat = softmax(index_mat);
  std::cout << "Softmax completed\n";
  //std::vector<T> v = softmax(std::vector<T>(10) = {10.0});






  return keys.data;
};

int main() {
  std::cout << "Hello, World!" << std::endl;
  //std::cout << attention()[0] << std::endl;
  //std::cout << attention()[1] << std::endl;


  std::vector<double> m_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> n_data = {2.0, 0.0, 0.0, 2.0};
  int rows = 2;
  int cols = 2;

  Matrix<double> m = Matrix<double>(rows, cols);
  Matrix<double> n = Matrix<double>(rows, cols);
  m.populate(m_data);
  n.populate(n_data);
  m.print();
  n.print();
  m.transpose();
  m.print();
  m / 2;
  m.print();
  Matrix<double> c = Matrix<double>(rows, cols);
  c.print();
  c = matmul(m, n);
  c.print();
  attention<double>(m, n, c);

  std::default_random_engine device;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);




  return 0;
}
