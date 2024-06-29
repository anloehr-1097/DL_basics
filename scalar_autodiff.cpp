#include <iostream>
#include <cmath>


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
    Value* preds = nullptr;
    Op op; 
    double grad = 0.0;

    Value(double val, Value *_preds, int _n_preds, Op _op) {
        data = val;
        preds = _preds;
        op = _op;
        n_preds = _n_preds;
    }

    void backward(double *grads) {
        // make sure to only call on root node
        if (grads == nullptr){
            grad = 1.0;
            for (int i = 0; i < n_preds; i++){
                preds[i].backward(&grad);
            }

        }



    }


    void print(){
        std::cout << "Data: ";
        std::cout << data << std::endl;
        std::cout << "Op: ";
        if (op == Op::ADD) std::cout << "Add" << std::endl;
        if (op == Op::SUB) std::cout << "Sub" << std::endl;
        if (op == Op::MUL) std::cout << "MUL" << std::endl;
        if (op == Op::EXP) std::cout << "EXP" << std::endl;
        if (op == Op::SIN) std::cout << "SIN" << std::endl;
        if (op == Op::NOOP) std::cout << "No op (leaf)" << std::endl;
        std::cout << "Pred values: ";
        for (int i = 0; i < n_preds; i++){
            std::cout << preds[i].data << " ";
        }

        std::cout << "Grad:" << std::endl;
        std::cout << grad << std::endl;


    }



};


int main(){


    // exp((a + b) * c) 
    Value a(1.0, nullptr, 0, Op::NOOP);
    Value b(2.0, nullptr, 0, Op::NOOP);
    Value c(3.0, nullptr, 0, Op::NOOP);

    Value y1_pred[2] = {a, b};
    Value y1(a.data + b.data, y1_pred, 2, Op::ADD);
   
    Value y2_pred[2] = {y1, c};
    Value y2(y1.data * c.data, y2_pred, 2, Op::MUL);

    Value y3_pred[1] = {y2};
    Value y3(exp(y2.data), y3_pred, 1, Op::EXP);

    a.print();
    b.print();
    c.print();
    y1.print();
    y2.print();
    y3.print();


};

