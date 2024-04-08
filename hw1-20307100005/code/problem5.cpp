#include <iostream>
#include <tuple>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pair;

/* Utility functions */

bool checkOrthogonality(MatrixXd a){
  int n = a.cols();
  for (int i = 0; i < n; i++){
    for (int j = i+1; j < n; j++){
      if (a.col(i).dot(a.col(j)) > 1e-10){
        return false;
      }
    } 
  }
  return true;
}

/* Implement the modified Gram-Schmidt algorithm. */

pair<MatrixXd,MatrixXd> ModifiedGramSchmidt(MatrixXd A)
{
  // MGS w/ column pivoting. 

  int n = A.cols();
  Eigen::MatrixXd R(n, n);

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

/* Test the MGS algorithm. */

void testMGS(MatrixXd m){
  /*
  Test the QR factorization using modified Gram-Schmidt. Criterion:

  1. The product of Q and R should be equal to the input matrix. You can compare the output QR with the input matrix.

  2. Q should be orthonormal. If so, the function checkOrthogonality will print "Q's orthonormality: 1".

  Things will be printed in the following order:
  1. The input matrix A.
  2. The output matrix Q.
  3. The output matrix R.
  4. The result of QR.
  */

  std::cout << "######################################" << std::endl << "########## A NEW TEST START ##########" << std::endl << "######################################" << std::endl << std::endl;
  std::cout << "---- The tested input ----" << std::endl << m << std::endl;

  pair<MatrixXd,MatrixXd> result = ModifiedGramSchmidt(m);
  m = result.first;
  MatrixXd R = result.second;

  std::cout << "---- Output Q ----" << std::endl << m << std::endl;
  std::cout << "---- Output R ----" << std::endl << R << std::endl;
  std::cout << "---- Output QR ----" << std::endl << m * R << std::endl;
  std::cout << "Q's orthonormality: " << checkOrthogonality(m) << std::endl << std::endl;
}


int main(){

  // Random square matrix.
  testMGS(MatrixXd::Random(3,3) * 50);

  // Low-rank square matrix.
  MatrixXd m(4,4);
  m << 1,-1,0,4,1,4,2,-2,1,4,2,2,1,-1,0,0;  
  testMGS(m);

  // Random rectangular matrix.
  testMGS(MatrixXd::Random(5,3) * 50);

}










    // TO DO: Debug the column pivoting part. What the hell happened here?

    /* BEGIN OF DEBUG PART

    int maxIndex = j;
    double maxNorm = A.col(j).norm();
    for (int i = j+1; i < n; i++){
      if (A.col(i).norm() > maxNorm){
        maxIndex = i;
        maxNorm = A.col(i).norm();
      }
    }

    A.col(j).swap(A.col(maxIndex));

    END OF DEBUG PART */
