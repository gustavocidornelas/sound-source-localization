import numpy as np


if __name__ == '__main__':
    C = np.genfromtxt('poly_fit_result_RIGHT.csv', skip_header=False, delimiter=',')

    # defining the transformation matrix
    T = np.array([[0, 1/3.0, -0.5, 1/6.0],
                  [0, -0.5, 0.5, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0]])

    # array that stores the transformed C vectors
    C_transformed = np.zeros((C.shape[0], C.shape[1]))

    # transforming each line
    for i in range(C.shape[0]):
        C_sample = C[i, :]

        # transforming to the canonical basis
        C_transformed[i, :] = np.dot(C_sample, T)
        print(i)

    # saving the transformed coefficients
    np.savetxt('transformed_poly_fit_result.csv', C_transformed, delimiter=',')
