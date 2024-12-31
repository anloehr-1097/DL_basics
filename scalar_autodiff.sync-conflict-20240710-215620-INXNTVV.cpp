#include <iostream>
#include <cmath>
#include <string>
#include <vector>


enum class Op {
    ADD,
    SUB,
    MUL,
    EXP,
    SIN,
    NOOP
};

class Value {

public:
    // public membeers of Value
    double data;
    int n_preds = 0;
    std::vector<Value> preds;
    Op op; 
    std::string name;
    double grad = 0.0;

    Value(double val, std::vector<Value>_preds, int _n_preds, Op _op, std::string _name) {
        data = val;
        preds = _preds;
        op = _op;
        n_preds = _n_preds;
        name = _name;
        grad = 0.0;
    }

    void backward(double child_grad, bool final){
      // make sure to only call on root node
      std::cout << "Backward called on node with value: " << data << std::endl;
      std::cout << "Child grad: " << child_grad << std::endl;
      std::cout << "No of preds: " << n_preds << std::endl;

      if (final == true){
        grad = 1.0;
      }
      else {
        grad += child_grad;
        std::cout << "Grad updated to: " << grad << std::endl;
      }
      
      for (int i = 0; i < n_preds; i++){
            std::cout << preds[i].data << std::endl;
        // logic given op here
        if (op == Op::ADD){
            // preds[i].grad += grad * preds[i].backward();
            preds[i].backward(grad * 1, false);
        }
        if (op == Op::SUB){
            preds[i].backward(grad * (-1), false);
        }
        if (op == Op::MUL){
            // multiplicy by other parent node
            if (i == 0){
                std::cout << "Multiplying by second parent " << preds[1].data << std::endl;
                preds[i].backward(grad * preds[1].data, false);
            }
            else {
                std::cout << "Multiplying by first parent " << preds[0].data << std::endl;
                preds[i].backward(grad * preds[0].data, false);
            }
        }
        if (op == Op::EXP){
            preds[i].backward(grad * exp(preds[i].data), false);
        }

        if (op == Op::SIN){
            preds[i].backward(grad * sin(preds[i].data), false);
        }

        if (op == Op::NOOP){
                ;
        }
    
    };
  }

    void print(){
        std::cout << "Name: ";
        std::cout << name << std::endl;
        std::cout << "Data: ";
        std::cout << data << std::endl;
        std::cout << "Op: ";
        if (op == Op::ADD) std::cout << "Add" << std::endl;
        if (op == Op::SUB) std::cout << "Sub" << std::endl;
        if (op == Op::MUL) std::cout << "MUL" << std::endl;
        if (op == Op::EXP) std::cout << "EXP" << std::endl;
        if (op == Op::SIN) std::cout << "SIN" << std::endl;
        if (op == Op::NOOP) std::cout << "No op (leaf)" << std::endl;
        std::cout << "Grad: " ;
        std::cout << grad << std::endl;
        std::cout << "Pred values: ";
        for (int i = 0; i < n_preds; i++){
            std::cout << preds[i].data << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;

    }

};


int main(){


    // exp((a + b) * c) 
    Value a(1.0, std::vector<Value>(), 0, Op::NOOP, std::string("a"));
    Value b(2.0, std::vector<Value>(), 0, Op::NOOP,std::string("b"));
    Value c(4.0, std::vector<Value>(), 0, Op::NOOP, std::string("c"));

    std::vector<Value> y1_pred = {a, b};
    Value y1(a.data + b.data, y1_pred, 2, Op::ADD, std::string("y1"));
   
    std::vector<Value> y2_pred = {y1, c};
    Value y2(y1.data * c.data, y2_pred, 2, Op::MUL, std::string("y2"));

    std::vector<Value> y3_pred = {y2};
    Value y3(exp(y2.data), y3_pred, 1, Op::EXP, std::string("y3"));

    // a.print();
    // b.print();
    // c.print();
    // y1.print();
    // y2.print();
    //y3.print();
    // y3.backward(1.0, true);

    y2.backward(1.0, true);
    a.print();
    b.print();
    c.print();
    y1.print();
    y2.print();

    // y3.print();


    // (a+b) * c
    // d/da = c = 3
    // d/db = c = 3
    // d/dc = a + b = 3
};

