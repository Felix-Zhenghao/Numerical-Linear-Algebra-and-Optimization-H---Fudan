# Test the correctness of the code

I implement problem 3,4,5 in one code file ```LU_PivotingAndSolve.cpp```.

When you run this file, all results will be save in a file called ```LUP_log.txt```. By running a test of an input matrix, the typical output written in that file will look as follows. Everything needed is printed. Moreover, since in problem 5 we don't need to solve any linear equations but only need to test the LU decomposition, you can set the parameter ```testSolver``` to ```false``` when running ```testLUP```. If you do so, any output related to solving linear equations will vanish.

```
######################################
########## A NEW TEST START ##########
######################################

---- The input matrix A ----
-49.8749  30.8741 -14.9709
 6.35853  8.50093  39.5962
-30.6696  -2.0127   32.284
---- The output matrix L ----
        1         0         0
   1.6262         1         0
-0.207324   0.23673         1
---- The output matrix U ----
-30.6696  -2.0127   32.284
       0  34.1471 -67.4711
       0        0  62.2619
---- The product of L and U ----
-30.6696  -2.0127   32.284
-49.8749  30.8741 -14.9709
 6.35853  8.50093  39.5962
---- The result of PA (should identical to LU) ----
-30.6696  -2.0127   32.284
-49.8749  30.8741 -14.9709
 6.35853  8.50093  39.5962

---- The solution vector x ----
 -1.36005
 -1.53203
-0.275723
---- The input vector b ----
 24.6605
-32.5892
 35.8943
---- The result of Ax (should identical to b) ----
 24.6605
-32.5892
 35.8943
```
By the way, **I feel sorry to print all information for the 100x100 matrix**. The output log can be long this time. Perhaps a concise version of checker can be implemented in the future.

# Test cases in the code

In the code, I generated four test cases: a square random matrix, a matrix used during the class to motivate partial column pivoting and two matrices required in problem 5.

If you choose to test the equation solver, please make sure that the linear system is solvable and have a unique solution because I didn't implement the solver for systems of infinite/no solutions.

I passed all these four cases on my laptop.


# Pay attention to use ABSOLUTE path to include ```Eigen/Dense```
I don't know why relative path doesn't work on my laptop. This is just a notification.


