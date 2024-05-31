#include <iostream>
#include <tuple>
#include <functional>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.


using Eigen::MatrixXd;
using Eigen::VectorXd;

// This file implement the LBFGS algorithm with fixed update stepsize.

// The objective function is f(x) = 1/2 x^T A x - b^T x.
// The gradient of f(x) is A x - b.
double objectiveFunction(VectorXd x, MatrixXd A, VectorXd b){
    VectorXd fx = 0.5 * x.transpose() * A * x - b.transpose() * x;
    return fx(0);
}


// The L-BFGS tow-loop recursion.
/**
 * @param x: the current x.
 * @param A: the matrix A in the objective function.
 * @param b: the vector b in the objective function.
 * @param S: the history of the difference of x.
 * @param Y: the history of the difference of the gradient.
 * @param stepsize: the fixed stepsize.
 * @param iteration: the current iteration count.
 * @return: the search direction.
*/
VectorXd recursion(VectorXd x, MatrixXd A, VectorXd b, 
                   MatrixXd S, MatrixXd Y, 
                   double stepsize, int iteration)
{
    // The gradient is A x - b.
    int historySize = S.cols();
    if (iteration<historySize){historySize = iteration;}
    VectorXd g = A * x - b;
    VectorXd p = -g;
    VectorXd alpha = VectorXd::Zero(historySize);

    // The first loop of the two-loop recursion.
    for (int i=historySize-1; i>=0; i--){
        alpha(i) = S.col(i).dot(p) / Y.col(i).dot(S.col(i));
        p = p - alpha(i) * Y.col(i);
    }
    p = p * Y.col(historySize-1).dot(S.col(historySize-1)) / Y.col(historySize-1).dot(Y.col(historySize-1));
    for (int i=0; i<historySize; i++){
        double beta = Y.col(i).dot(p) / Y.col(i).dot(S.col(i));
        p = p + S.col(i) * (alpha(i) - beta);
    }
    return p;
}


// The L-BFGS algorithm with fixed update stepsize.
/**
 * @param f: the objective function.
 * @param x: the initial guess.
 * @param A: the matrix A in the objective function.
 * @param b: the vector b in the objective function.
 * @param S: the history of the difference of x.
 * @param Y: the history of the difference of the gradient.
 * @param historySize: the size of the history.
 * @param stepsize: the fixed stepsize.
 * @param g_epsilon: the threshold of the gradient convergence.
 * @param delta: the threshold of the f convergence.
 * @return: the optimal x.
*/
VectorXd LBFGS(std::function<double(VectorXd x, MatrixXd A, VectorXd b)> f, 
             VectorXd x, MatrixXd A, VectorXd b, 
             MatrixXd S, MatrixXd Y, 
             int historySize, double stepsize, 
             double g_epsilon, double delta)
{

    // Initialize the history. S and Y has size n * historySize.
    S = MatrixXd::Zero(x.size(), historySize);
    Y = MatrixXd::Zero(x.size(), historySize);
    VectorXd g = A * x - b; // The gradient is A x - b.
    VectorXd p = -g; // The search direction.
    VectorXd g_delta; // The difference of the gradient.
    double f_delta; // The difference of the objective function.

    VectorXd x_old = x; // The old
    VectorXd g_old = g; // The old gradient.
    x = x + stepsize * p; // The updated x.
    g = A * x - b; // The updated gradient.
    S.col(0) = x - x_old;
    Y.col(0) = g - g_old;
    int iteration = 1; // The iteration counter.

    while (g_delta.norm() > g_epsilon && f_delta > delta){
        g_old = g;
        x_old = x;

        // get the search direction.
        p = recursion(x, A, b, S, Y, stepsize, iteration);
        // update x.
        x = x + stepsize * p;
        // update g.
        g = A * x - b;
        // update the history. From left to right, always the oldest to the newest history.
        if (iteration < historySize){
            S.col(iteration) = x - x_old;
            Y.col(iteration) = g - g_old;
        }
        else 
        {
            S = S.rightCols(historySize-1);
            Y = Y.rightCols(historySize-1);
            S.col(historySize-1) = x - x_old;
            Y.col(historySize-1) = g - g_old;
        }
        // update the difference.
        g_delta = g - g_old;
        f_delta = f(x, A, b) - f(x_old, A, b);
        // update the iteration.
        iteration++;
    }
    return x;
}



// test
int main(){
    // The objective function is f(x) = 1/2 x^T A x - b^T x.
    // The gradient of f(x) is A x - b.
    MatrixXd A(2, 2);
    A << 1, 0, 0, 1;
    VectorXd b(2);
    b << 1, 1;
    VectorXd x(2);
    x << 0, 0;
    MatrixXd S(2, 2);
    MatrixXd Y(2, 2);
    int historySize = 6;
    double stepsize = 0.1;
    double g_epsilon = 1e-5;
    double delta = 1e-6;
    VectorXd result = LBFGS(objectiveFunction, x, A, b, S, Y, historySize, stepsize, g_epsilon, delta)*10;
    std::cout << "The optimal x is: \n" << result <<  std::endl;
    std::cout << "The true optimal x is: \n" << A.inverse() * b <<  std::endl;
    return 0;
}

/* Output log:

    The optimal x is: 
    1
    1
    The true optimal x is:
    1
    1
    
*/



