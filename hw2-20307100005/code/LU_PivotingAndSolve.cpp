#include <iostream>
#include <fstream>
#include <tuple>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using std::pair;


std::ofstream file("LUP_log.txt"); // The result will be stored in this file.

// LU decomposition with partial pivoting.

pair<MatrixXd,VectorXi> LUPivoting(MatrixXd A){
    // We manipulate A in place to save memory. All information of L and U is stored in A.

    int m = A.rows();
    VectorXi p = VectorXi::LinSpaced(m, 0, m-1); // Vector containing the permutation history with initial values 0, 1, 2, ..., m-1.\

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

// Solve Ax = b using LU decomposition with partial pivoting.
VectorXd solveLUP(MatrixXd A, VectorXd b){
    // Solve Ax = b using LU decomposition with partial pivoting.
    // Suppose the equation is solvable and has unique solution.
    // We manipulate b in place to save memory. The solution is stored in b.

    // Get PA = LU.
    pair<MatrixXd,VectorXi> LU = LUPivoting(A);
    VectorXi p = LU.second;

    // To solve Ax=b, we solve PAx = Pb, that is LUx = Pb:

    // Pertube b according to the permutation history p.
    VectorXd temp = b;
    for (int i = 0; i < A.rows(); i++){
        b(i) = temp(p(i));
    }

    // Let y = Ux, then Ly = Pb. Solve Ly = Pb for y.
    int m = A.rows();
    for (int i = 0; i < m; i++){
        for (int j = 0; j < i; j++){
            b(i) -= LU.first(i,j) * b(j);
        }
    }

    // Let Ux = y. Solve Ux = y for x.
    for (int i = m-1; i >= 0; i--){
        for (int j = i+1; j < m; j++){
            b(i) -= LU.first(i,j) * b(j);
        }
        b(i) = b(i) / LU.first(i,i);
    }

    return b;

}


// Test the LU decomposition and solve functions.
void testLUP(MatrixXd A, VectorXd b, bool testSolver = true){
    /*
    Test the LU decomposition with partial pivoting. Criterion:

    1. The product of L and U should be equal to the input matrix. You can compare the output LU with the input matrix.
    3. The solution x should make Ax = b. Ax will be printed to check whether it is equal to b.

    Things will be printed in the following order:
    1. The input matrix A.
    2. The output matrix L.
    3. The output matrix U.
    4. The product of L and U to check whether LU=A.
    5. The result of PA to check whether PA=LU.
    6. The solution vector x.
    7. The input vector b.
    8. The result of Ax to check whether Ax=b.
    */

    file << "######################################" << std::endl << "########## A NEW TEST START ##########" << std::endl << "######################################" << std::endl << std::endl;
    if (!testSolver){
        file << std::endl << "######## This test only tests the LU decomposition of the matrix ########" << std::endl << A << std::endl;
    }
    file << "---- The input matrix A ----" << std::endl << A << std::endl;

    // LU decomposition with partial pivoting.
    pair<MatrixXd,VectorXi> LU = LUPivoting(A);
    MatrixXd L = LU.first.triangularView<Eigen::UnitLower>(); // Extract the lower triangular part of LU and set the diagonal to 1.
    MatrixXd U = LU.first.triangularView<Eigen::Upper>(); // Extract the upper triangular part of LU.
    VectorXi p = LU.second; // The permutation history.
    file << "---- The output matrix L ----" << std::endl << L << std::endl;
    file << "---- The output matrix U ----" << std::endl << U << std::endl;
    file << "---- The product of L and U ----" << std::endl << L * U << std::endl;
    
    // Get PA according to the permutation history p.
    MatrixXd PA = MatrixXd(A.rows(),A.cols());
    for (int i = 0; i < A.rows(); i++){
        PA.row(i) = A.row(p(i));
    }
    file << "---- The result of PA (should identical to LU) ----" << std::endl << PA << std::endl << std::endl;

    if (!testSolver){
        return; // You can choose to test only the LU decomposition part of the implementation. (For problem 5)
    }

    // Solve Ax = b.
    VectorXd x = solveLUP(A,b);
    file << "---- The solution vector x ----" << std::endl << x << std::endl;
    file << "---- The input vector b ----" << std::endl << b << std::endl;
    file << "---- The result of Ax (should identical to b) ----" << std::endl << A * x << std::endl;
    
}

MatrixXd generateTestCase(int n){
    // Generate the test case of problem 5.

    MatrixXd A = MatrixXd::Ones(n,n);
    // set the lower triangular part of A to -1.
    A.triangularView<Eigen::StrictlyLower>() = MatrixXd::Constant(n,n,-1);

    return A;
}


int main(){

    // Random square matrix.
    MatrixXd A = MatrixXd::Random(3,3) * 50;
    VectorXd b = VectorXd::Random(3) * 50;
    testLUP(A,b);


    // Cases given during the class to motivate partial column pivoting.
    MatrixXd A1(4,4);
    A1 << 2,1,1,0,
        4,3,3,1,
        8,7,9,5,
        6,7,9,8;
    VectorXd b1(4);
    b1 << 4,7,23,32;
    testLUP(A1,b1);

    MatrixXd A2 = generateTestCase(8);
    VectorXd b2 = VectorXd::Ones(8);
    testLUP(A2,b2,false); // Test the LU decomposition only.

    MatrixXd A3 = generateTestCase(100);
    VectorXd b3 = VectorXd::Ones(100);
    testLUP(A3,b3,false); // Test the LU decomposition only.

    return 0;

}
