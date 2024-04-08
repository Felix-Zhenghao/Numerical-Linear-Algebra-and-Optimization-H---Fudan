#include <iostream>
#include <tuple>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pair;


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


bool checkOrthogonality(MatrixXd a){
  // This func will be used to test the orthogonality of Q.

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

/* 
Implement the Householder QR factorizaiton.
To prevent namespace confict with Eigen library, I name the func as HQR.
*/

pair<MatrixXd,MatrixXd> HQR(MatrixXd A, bool returnQ = true){

  // Full QR factorization using Householder reflectors w/ column pivoting.
  // The input matrix A will be manipulated in place to get R.
  // If returnQ is true, the function will also return Q.

  int n = A.cols();
  int m = A.rows();

  MatrixXd Q = MatrixXd::Identity(m,m); // To prevent error I allocate memory for Q anyway.

  for (int k = 0; k < n; k++){
    // To save memory, manipulate A in-place to get R.

    VectorXd x = A.col(k).tail(m-k);
    std::cout << "---- Output Q ----" << std::endl << x.size() << std::endl;
    VectorXd v = VectorXd::Zero(m-k); // This can be optimized to use a single vector for all k

    // Find the direction to be projected out.
    v(0) = x.norm()*sign_func(x(0));
    v += x;
    v = v.normalized();

    A.bottomRightCorner(m-k,n-k) -= 2 * v * (v.transpose() * A.bottomRightCorner(m-k,n-k));

    if (returnQ){
      // When Q is needed, also manipulate Q in place.
      // Only need to change bottom m-k rows of Q when multiplying Q by a householder reflector.
      Q.bottomRows(m-k) -= 2 * v * (v.transpose() * Q.bottomRows(m-k));
    }

  }
    
  if (returnQ){
    Q = Q.transpose().eval();
    return std::make_pair(Q,A);
  }

  else{return std::make_pair(A,A);} // Only return A if Q is not needed.
  
}

/* Test the Householder QR factorization. */

void testHouseholderQR(MatrixXd m){

  /*
  Test the Householder QR factorization. Criterion:

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

  pair<MatrixXd,MatrixXd> result = HQR(m);
  m = result.first;
  MatrixXd R = result.second;

  std::cout << "---- Output Q ----" << std::endl << m << std::endl;
  std::cout << "---- Output R ----" << std::endl << R << std::endl;
  std::cout << "---- Output QR ----" << std::endl << m * R << std::endl;
  std::cout << "Q's orthonormality: " << checkOrthogonality(m) << std::endl << std::endl;
}


int main(){

  // Random square matrix.
  testHouseholderQR(MatrixXd::Random(3,3) * 50);

  // Low-rank square matrix.
  MatrixXd m(4,4);
  m << 1,-1,0,4,1,4,2,-2,1,4,2,2,1,-1,0,0;  
  testHouseholderQR(m);

  // Random rectangular matrix.
  testHouseholderQR(MatrixXd::Random(5,3) * 50);

}
