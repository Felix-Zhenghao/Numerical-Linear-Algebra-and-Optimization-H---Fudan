# Test the correctness of the code

For problem 5 (MGS) and problem 6 (Housholder QR), I use function called ```testMGS``` and ```testHQR``` to ```print``` all metrics needed to verify the algorithm implementation.

For example, the output of the ```testMGS``` when running ```C++ testMGS(MatrixXd::Random(3,3) * 50)``` can look as follows. From the output log, we can see that QR is 'identical' to A, Q is orthonormal and R is an upper-triangular matrix.

```

######################################
########## A NEW TEST START ##########
######################################

---- The tested input ----
-49.8749  30.8741 -14.9709
 6.35853  8.50093  39.5962
-30.6696  -2.0127   32.284
---- Output Q ----
-0.846852  0.492958 -0.199586
 0.107965  0.526819  0.843093
-0.520755 -0.692427  0.499359
---- Output R ----
     58.8945     -24.1798     0.141044
3.62662e-317      21.0917     -8.87427
           0            0      52.4926
---- Output QR ----
-49.8749  30.8741 -14.9709
 6.35853  8.50093  39.5962
-30.6696  -2.0127   32.284
Q's orthonormality: 1

```

Typically, I test the algorithm with a randomly-generated square matrix, a randomly-generated rectangular matrix and a low-rank square matrix (cases given to motivate column pivoting during the class). I successfully passed all the tests on my laptop. However, I actually didn't implement the column pivoting part. For the case given during the class, I simply passed the test by exploiting the high-precision datatype ```VecotrXd``` or ```MatrixXd```.

# Pay attention to use ABSOLUTE path to include ```Eigen/Dense```
I don't know why relative path doesn't work on my laptop. This is just a notification.

# Workflow

File ```problem5.cpp``` and ```problem6.cpp``` is self-contained and can be ran independently. To plot the result of problem 8, you have to run ```problem8.cpp``` first then run ```problem8_plot.py```.

















