// Implement QR algorithm for the eigenvalue problem of hermitian matrices with wilkinson shift.

#include <iostream>
#include <fstream>
#include <tuple>
#include <D:\\CSAPP\Eigen\Dense> // Please import the Eigen library before running the code.

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using std::pair;
using std::min;
using std::abs;


std::ofstream file("QRHermitian_log.txt"); // The result will be stored in this file.



// get the QR decomposition of A.
// Eigen::HouseholderQR<MatrixXd> qr(A);
// MatrixXd Q = qr.householderQ();
// MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();



// Implement QR algorithm for the eigenvalue problem of hermitian matrices with wilkinson shift.
pair<VectorXd,MatrixXd> QR_Hessenberg(MatrixXd A){

    // Initialize the eigenvalues and eigenvectors.
    int n = A.rows();
    MatrixXd Eigenvectors = MatrixXd::Identity(n,n);
    VectorXd Eigenvalues = VectorXd::Zero(n);

    // Reduce A to Hessenberg form.
    Eigen::HessenbergDecomposition<MatrixXd> hess(A);
    MatrixXd H = hess.matrixH();
    MatrixXd Q_ = hess.matrixQ();

    // QR iteration.
    for (int i = 0; i < 50; i++){
        // Get the eigenvalues of H_end: a*b = det(H_end), a+b = trace(H_end).
        Eigen::Matrix2d H_end = H.block(n-2,n-2,2,2);
        double determinant = H_end(0,0) * H_end(1,1) - H_end(0,1) * H_end(1,0);
        double trace = H_end(0,0) + H_end(1,1);
        double eigen1 = (trace + sqrt(trace*trace - 4*determinant)) / 2;
        double eigen2 = (trace - sqrt(trace*trace - 4*determinant)) / 2;
        double shift = abs(eigen1 - H_end(1,1)) < abs(eigen2 - H_end(1,1)) ? eigen1 : eigen2;

        // Perform the wilkinson shift.
        H = H - shift * MatrixXd::Identity(n,n);

        // Get the QR decomposition of H.
        Eigen::HouseholderQR<MatrixXd> qr(H);
        MatrixXd Q = qr.householderQ();
        MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

        // Update H.
        H = R * Q + shift * MatrixXd::Identity(n,n);

        // Update the eigenvectors.
        Eigenvectors = Eigenvectors * Q;
    }

    Eigenvalues = H.diagonal();
    Eigenvectors = Q_ * Eigenvectors;
    return std::make_pair(Eigenvalues,Eigenvectors);
}



// Test the result of eigenvalues and eigenvectors.
void testEigen(MatrixXd A){
    /*
    Test QR algorithm for the eigenvalue problem of hermitian matrices. Criterion:

    1. Eigenvalues and eigenvectors are the same as those computed by Eigen library.

    Things will be printed in the following order:
    0. The input Hermitian matrix A.
    1. Eigen library's result of eigenvalues.
    2. Eigen library's result of eigenvectors.
    3. My result of eigenvalues.
    4. My result of eigenvectors.
    */

    file << "######################################" << std::endl << "########## A NEW TEST START ##########" << std::endl << "######################################" << std::endl << std::endl;
    file << "---- The input matrix A ----" << std::endl << A << std::endl;

    // Get the result of Eigen library.
    Eigen::EigenSolver<MatrixXd> es(A);
    file << "---- Eigen library's result of eigenvalues ----" << std::endl << es.eigenvalues() << std::endl;
    file << "---- Eigen library's result of eigenvectors ----" << std::endl << es.eigenvectors() << std::endl;
    // Get the result of my implementation.
    pair<VectorXd,MatrixXd> result = QR_Hessenberg(A);
    file << "---- My result of eigenvalues ----" << std::endl << result.first << std::endl;
    file << "---- My result of eigenvectors ----" << std::endl << result.second << std::endl << std::endl;
}


int main(){
    MatrixXd A = MatrixXd::Random(10,10);
    A = A * A.transpose(); // Make A symmetric positive definite.
    testEigen(A);

    return 0;
}
