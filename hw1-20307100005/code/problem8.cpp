#include <iostream>
#include <fstream>
#include <tuple>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::HouseholderQR;
using std::pair;

/*
The general workflow of problem 8 is:

1. Run this file to generate the rjj values for the three methods.
2. This file will store the rjj's in a file called rjj.txt.
3. Run the python file 'plot.py' to plot the rjj values.
*/

double sign_func(double x)
{
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}

/* Classical GS */

pair<MatrixXf,MatrixXf> ClassicalGramSchmidt(MatrixXf A)
{
  // Classical GS w/ column pivoting. 

  int n = A.cols();
  Eigen::MatrixXf R(n, n);

  // Manipulate A in place to save memory.
  for (int j = 0; j < n; j++){
    for (int k = 0; k < j; k++){
        R(k,j) = A.col(j).dot(A.col(k));
        A.col(j) = A.col(j) - R(k,j) * A.col(k);
    }

    R(j,j) = A.col(j).norm();
    A.col(j) = A.col(j) / R(j,j);
  }
    return std::make_pair(A,R);
}

/* MGS */
pair<MatrixXf,MatrixXf> ModifiedGramSchmidt(MatrixXf A)
{
  // MGS w/ column pivoting. 

  int n = A.cols();
  Eigen::MatrixXf R(n, n);

  // Manipulate A in place to save memory.
  for (int j = 0; j < n; j++){

    R(j,j) = A.col(j).norm();
    A.col(j) = A.col(j).normalized();

    for (int k = j+1; k < n; k++){
      R(j,k) = A.col(j).dot(A.col(k));
      A.col(k) = A.col(k) - R(j,k) * A.col(j);
    }

  }
    return std::make_pair(A,R);
}

/* Housholder QR */
pair<MatrixXf,MatrixXf> HQR(MatrixXf A, bool returnQ = true){

  int n = A.cols();
  int m = A.rows();

  MatrixXf Q = MatrixXf::Identity(m,m); // To prevent error I allocate memory for Q anyway.

  for (int k = 0; k < n; k++){

    VectorXf x = A.col(k).tail(m-k);
    VectorXf v = VectorXf::Zero(m-k); // This can be optimized to use a single vector for all k

    v(0) = x.norm()*sign_func(x(0));
    v += x;
    v = v.normalized();

    A.bottomRightCorner(m-k,n-k) -= 2 * v * (v.transpose() * A.bottomRightCorner(m-k,n-k));

    if (returnQ){
      Q.bottomRows(m-k) -= 2 * v * (v.transpose() * Q.bottomRows(m-k));
    }

  }
    
  if (returnQ){
    Q = Q.transpose().eval();
    return std::make_pair(Q,A);
  }

  else{return std::make_pair(A,A);} // Only return A if Q is not needed.
  
}


MatrixXf generateExponetiallyDegradedSingularValueMatrix(int m, int n, double decay_rate)
{
    int k = std::min(m,n);
    VectorXf singularValues(k);

    for (int i = 0; i < k; ++i) {
        singularValues(i) = std::pow(decay_rate, i);
    }

    MatrixXf A = MatrixXf::Zero(m, n);
    A.topLeftCorner(k,k) = singularValues.asDiagonal();

    // Generate random matrices U and V
    // Then use QR decomposition of random matrices to obtain orthogonal matrices
    MatrixXf randomMatrixU = MatrixXf::Random(m, m) * 0.5;
    HouseholderQR<MatrixXf> qrU(randomMatrixU);
    MatrixXf matrixU = qrU.householderQ();
    
    MatrixXf randomMatrixV = MatrixXf::Random(n, n) * 0.5;
    HouseholderQR<MatrixXf> qrV(randomMatrixV);
    MatrixXf matrixV = qrV.householderQ();

    // Construct the matrix A with exponentially decaying singular values
    A = matrixU * A * matrixV.transpose();
    return A;
}


int main(){

    // Generate a matrix with exponentially decaying singular values.
    MatrixXf A = generateExponetiallyDegradedSingularValueMatrix(500, 500, 0.6);

    // Use classical GS to factorize the matrix A and get the R.
    pair<MatrixXf,MatrixXf> resultCGS = ClassicalGramSchmidt(A);
    MatrixXf R_CGS = resultCGS.second;
    VectorXf rjj_CGS = R_CGS.diagonal();
    std::sort(rjj_CGS.data(), rjj_CGS.data() + rjj_CGS.size(), std::greater<double>()); // Sort the diagonal elements in descending order.

    // Use MGS to factorize the matrix A and get the R.
    pair<MatrixXf,MatrixXf> resultMGS = ModifiedGramSchmidt(A);
    MatrixXf R_MGS = resultMGS.second;
    VectorXf rjj_MGS = R_MGS.diagonal();
    std::sort(rjj_MGS.data(), rjj_MGS.data() + rjj_MGS.size(), std::greater<double>()); // Sort the diagonal elements in descending order.

    // Use Housholder QR to factorize the matrix A and get the R.
    pair<MatrixXf,MatrixXf> resultHQR = HQR(A);
    MatrixXf R_HQR = resultHQR.second;
    VectorXf rjj_HQR = R_HQR.diagonal();
    rjj_HQR = rjj_HQR.array().abs(); // The diagonal elements of R by HQR are not guaranteed to be positive. So take the absolute value.
    std::sort(rjj_HQR.data(), rjj_HQR.data() + rjj_HQR.size(), std::greater<double>()); // Sort the diagonal elements in descending order.

    // Write to a file
    std::ofstream file("rjj.txt");
    if (file.is_open()) {
        file << rjj_CGS << std::endl;
        file << rjj_MGS << std::endl;
        file << rjj_HQR << std::endl;
        file.close();
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 1;
    }

    std::cout << "rjj's written to file." << std::endl;

  return 0;
}














