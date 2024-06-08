#include <iostream>
#include <fstream>
#include <tuple>
#include <functional>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.


using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;
using std::pair;

std::ofstream file("LBFGS_log.txt");

double objectiveFunction(VectorXd x, MatrixXd A, VectorXd b){
    VectorXd fx = 0.5 * x.transpose() * A * x - b.transpose() * x;
    return fx(0);
}

VectorXd gradient(VectorXd x, MatrixXd A, VectorXd b){
    return A * x - b;
}



VectorXd get_direction(MatrixXd S,MatrixXd Y,
                       VectorXd g, int historySize,
                       int newest)
{
    VectorXd p = -g;
    VectorXd alpha = VectorXd::Zero(historySize);

    // newest is the col index of the newest history
    for (int i = 0; i < historySize; i++)
    {
        // loop from the newest to the oldest
        int index = newest - i;
        if (index < 0)
        {
            index += historySize;
        }

        alpha(index) = S.col(index).dot(p) / S.col(index).dot(Y.col(index));
        p -= alpha(index) * Y.col(index);
    }

    p *= S.col(newest).dot(Y.col(newest)) / Y.col(newest).dot(Y.col(newest));

    for (int i = 0; i < historySize; i++)
    {
        // loop from the oldest to the newest
        int index = newest + 1 + i ;
        if (index >= historySize)
        {
            index -= historySize;
        }

        double beta = Y.col(index).dot(p) / S.col(index).dot(Y.col(index));
        p += (alpha(index) - beta) * S.col(index);
    }

    return p;
}





/*
* @param x: the initial guess.
* @param history_size: the size of the history.
* @param convergence_threshold: the norm of the difference of x when converged.
* @param A: the matrix A in the objective function.
* @param b: the vector b in the objective function.
*/
pair<VectorXd,VectorXd> LBFGS(VectorXd x, int history_size,
               double convergence_threshold,
                MatrixXd A, VectorXd b)
{
    double x_diff = 1.0;
    int newest = 0;
    int historySize = 1;
    int iter = 0;
    VectorXd x_history = VectorXd::Zero(5000);
    MatrixXd S = MatrixXd::Zero(x.size(), history_size);
    MatrixXd Y = MatrixXd::Zero(x.size(), history_size);

    double stepsize = 1;
    double rho = 0.8;
    double c = 0.1;
    VectorXd optimal = A.inverse() * b;

    VectorXd g = gradient(x, A, b);
    VectorXd x_new = x+0.01*g;
    VectorXd g_new = gradient(x_new, A, b);
    S.col(newest) = x_new - x;
    Y.col(newest) = g_new - g;

    x = x_new;

    while (x_diff > convergence_threshold)
    {
        // calculate gradient
        g = gradient(x, A, b);

        // calculate search direction
        VectorXd p = get_direction(S,Y,g,historySize,newest); // TO DO

        // get the stepsize using backtracking line search
        while (objectiveFunction(x + stepsize*p, A, b) > objectiveFunction(x, A, b) + c*stepsize*g.dot(p))
        {
            stepsize *= rho;
        }

        // update x
        x_new = x + stepsize*p;
        x_diff = (x_new - x).norm();

        // update history
        newest = (newest + 1) % history_size;
        g_new = gradient(x_new, A, b);
        S.col(newest) = x_new - x;
        Y.col(newest) = g_new - g;
        x_history(iter) = (x - optimal).norm();
        
        x = x_new;
        if (historySize < history_size){historySize++;}
        iter++;
    }

    return std::make_pair(x,x_history.head(iter));
}


void test(int col){
    // The objective function is f(x) = 1/2 x^T A x - b^T x.
    // The gradient of f(x) is A x - b.

    MatrixXd A = MatrixXd::Random(col,col);
    A = A * A.transpose();
    VectorXd b = VectorXd::Random(col);
    VectorXd x(col);
    x = VectorXd::Zero(col);
    int historySize = 6;
    double delta = 1e-9;

    pair<VectorXd,VectorXd> result = LBFGS(x, historySize, delta, A, b);
    file << "The optimal x is: \n" << result.first <<  std::endl;
    file << "The update history is: \n" << result.second <<  std::endl;
    file << "The true optimal x is: \n" << A.inverse() * b <<  std::endl;

}


int main()
{
    test(5);
    test(10);
    test(15);
    test(20);
    test(100);
    return 0;
}




