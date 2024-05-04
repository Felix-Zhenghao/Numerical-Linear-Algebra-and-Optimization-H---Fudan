// Implement Householder reduction to Hessenberg form for Hermitian matrices.

#include <iostream>
#include <fstream>
#include <tuple>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using std::pair;


std::ofstream file("Hessenberg_log.txt"); // The result will be stored in this file.


/* Utility functions */
double sign_func(double x)
{
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}



pair<MatrixXd,MatrixXd> GetHessenberg(MatrixXd A){
    // Fully exploit the Hermitian property.
    // A is a Hermitian matrix with size of (n,n)

    int n = A.cols();

    MatrixXd Q = MatrixXd::Identity(n,n); // To prevent error I allocate memory for Q anyway.

    for (int k = 0; k < n-1; k++){
        /* In the k-th iteration A[n-k:,n-k:] is split into four parts:
        *      _       _
        *     |  a   b  |
        *     |_ c   d _|
        * 
        *  In the k-th iteration: 
        *  - size_a = (1)*(1), size_b = (1)*(n-k-1), size_c = (n-k-1)*(1), size_d = (n-k-1)*(n-k-1)
        *  - a is not changed, c is left multiplied by householder reflector, 
        *  - c is copied from b to exploit the Hermitian property, d is both left and right multiplied by householder reflector.
        *  - a = A[k,k], b = A[k,k+1:n-1], c = A[k+1:n-1,k], d = A[k+1:n-1,k+1:n-1]
        */
        
        VectorXd c = A.col(k).tail(n-k-1); // c = A[k+1:n-1,k]
        VectorXd v = VectorXd::Zero(n-k-1); // This can be optimized to use a single vector for all k

        // Find the direction to be projected out.
        v(0) = c.norm()*sign_func(c(0));
        v += c;
        v = v.normalized();

        // Left multiply the householder reflector with c and d. [c d] = A[k+1:n,k:n]
        A.bottomRightCorner(n-k-1,n-k) -= 2 * v * (v.transpose() * A.bottomRightCorner(n-k-1,n-k));

        // Copy c to b to exploit the Hermitian property.
        A.row(k).tail(n-k-1) = A.col(k).tail(n-k-1).transpose();

        // Right multiply the householder reflector with d.
        // Because we exploited the Hermitian property, this step manipulates A[k+1:n-1,k+1:n-1] rather than A[0:n-1,k+1:n-1]
        A.bottomRightCorner(n-k-1,n-k-1) -= 2 * (A.bottomRightCorner(n-k-1,n-k-1) * v) * v.transpose();

        // Manipulate Q in place.
        Q.bottomRows(n-k-1) -= 2 * v * (v.transpose() * Q.bottomRows(n-k-1));
    }

    Q = Q.transpose().eval();
    return std::make_pair(Q,A);
}


// Test the LDLT decomposition and solve functions.
void testHessenberg(MatrixXd A){
    /*
    Test the householder reduction to Hessenberg form for Hermitian matrix. Criterion:

    1. The QHQ* should be equal to the input matrix. You can compare the output QHQ* with the input matrix.

    Things will be printed in the following order:
    1. The input matrix A.
    2. The output matrix Q.
    3. The output matrix H.
    4. The product of QHQ* to check whether QHQ*=A.
    */

    file << "######################################" << std::endl << "########## A NEW TEST START ##########" << std::endl << "######################################" << std::endl << std::endl;
    file << "---- The input matrix A ----" << std::endl << A << std::endl;

    // LU decomposition with partial pivoting.
    pair<MatrixXd,MatrixXd> HH = GetHessenberg(A);
    MatrixXd Q = HH.first; // Extract the lower triangular part of LU and set the diagonal to 1.
    MatrixXd H = HH.second; // The diagonal matrix.
    file << "---- The output matrix Q ----" << std::endl << Q << std::endl;
    file << "---- The output matrix H ----" << std::endl << H << std::endl;
    file << "---- The result of QHQ* (this should equal to A)----" << std::endl << Q * H * Q.transpose() << std::endl;
}

// Test the result of Hessenberg reduced form using a random 10x10 matrix.

int main(){
    MatrixXd A = MatrixXd::Random(10,10);
    A = A * A.transpose(); // Make A symmetric positive definite.
    testHessenberg(A);

    return 0;
}


