"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset_tf(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1],
                                                     [1, x2],
                                                     [1, x3],
                                                     .......]
                                where xi is the 16-dimensional feature of each sample

        T(numpy.ndarray): class label vector T = [[y1],
                                                  [y2],
                                                  [y3],
                                                   ...]
                             where yi is 1/0, the label of each sample
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    A = []
    T = []

    # reading index file
    indexf = open(path_to_dataset_folder+'/'+index_filename,'r')
    for indxline in indexf:
        temp = indxline.split()

        # Label append
        if (int(temp[0]) == 1):
            T.append(1)
        elif (int(temp[0]) == -1):
            T.append(0)

        # read the feature file
        for line in open(path_to_dataset_folder+'/'+temp[1],'r'):
            A.append(line.split())

    # Formatting
    T = np.array(T)
    A = np.array(A)
    T = T.astype(np.float64)
    A = A.astype(np.float64)
    A = np.concatenate((np.ones((A.shape[0],1)), A), axis=1)
    T = np.reshape(T,(T.shape[0],1))

    return A, T
