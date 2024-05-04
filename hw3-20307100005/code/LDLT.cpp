// Implement LDLT factorization with col pivoting and test with 10x10 random matrix.

#include <iostream>
#include <fstream>
#include <tuple>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using std::pair;


std::ofstream file("LDLT_log.txt"); // The result will be stored in this file.

// LDLT decomposition with partial pivoting.
pair<MatrixXd,VectorXi> LDLTPivoting(MatrixXd A){
    // We manipulate A in place to save memory. All information of L is stored in A.

    int m = A.rows();
    VectorXi p = VectorXi::LinSpaced(m, 0, m-1); // Vector containing the permutation history with initial values 0, 1, 2, ..., m-1.

    for (int k = 0; k < m-1; k++){

        // Find the pivot.
        int pivot = k;
        double max = A(k,k);
        for (int i = k+1; i < m; i++){
            if (std::abs(A(i,k)) > max){
                max = std::abs(A(i,k));
                pivot = i;
            }
        }

        // Swap rows and update the permutation history.
        if (pivot != k){
            A.row(k).swap(A.row(pivot)); // By swapping the rows, we are also updating the lower triangular part of A.

            double temp = p(k); 
            p(k) = p(pivot);
            p(pivot) = temp;
        }

        // Eliminate the lower triangular part and store the multipliers in the lower triangular part.
        for (int i = k+1; i < m; i++){
            A(i,k) = A(i,k) / A(k,k);
            A.block(i,k+1,1,m-k-1) -= A(i,k) * A.block(k,k+1,1,m-k-1);
        }

    }

    return std::make_pair(A,p);

}




// Test the LDLT decomposition and solve functions.
void testLUP(MatrixXd A){
    /*
    Test the LDLT decomposition with partial pivoting. Criterion:

    1. The LDL^T should be equal to the input matrix. You can compare the output LDL^T with the input matrix.
    2. The permutation history p should be correct.

    Things will be printed in the following order:
    1. The input matrix A.
    2. The output matrix L.
    3. The output matrix D.
    4. The product of LDL^{T} to check whether LDL^{T}=A.
    5. The result of PA to check whether PA=LU.
    */

    file << "######################################" << std::endl << "########## A NEW TEST START ##########" << std::endl << "######################################" << std::endl << std::endl;
    file << "---- The input matrix A ----" << std::endl << A << std::endl;

    // LU decomposition with partial pivoting.
    pair<MatrixXd,VectorXi> LDLT = LDLTPivoting(A);
    MatrixXd L = LDLT.first.triangularView<Eigen::UnitLower>(); // Extract the lower triangular part of LU and set the diagonal to 1.
    file << L.size() << "CHECK" << std::endl;
    MatrixXd D = LDLT.first.diagonal().asDiagonal(); // The diagonal matrix.
    VectorXi p = LDLT.second; // The permutation history.
    file << "---- The output matrix L ----" << std::endl << L << std::endl;
    file << "---- The output matrix D ----" << std::endl << D << std::endl;
    file << "---- The result of LDL^{T} (this should equal to A)----" << std::endl << L * D * L.transpose() << std::endl;
    
    // Get PA according to the permutation history p.
    MatrixXd PA = MatrixXd(A.rows(),A.cols());
    for (int i = 0; i < A.rows(); i++){
        PA.row(i) = A.row(p(i));
    }
    file << "---- The result of PA (should identical to LDL^T) ----" << std::endl << PA << std::endl << std::endl;
}



// Test the result of LDLT decomposition using a random 10x10 matrix.

int main(){
    MatrixXd A = MatrixXd::Random(10,10);
    A = A * A.transpose(); // Make A symmetric positive definite.
    testLUP(A);

    return 0;
}








