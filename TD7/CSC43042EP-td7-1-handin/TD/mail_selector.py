import numpy as np
from TD.svm import SVM
from TD.kernel import Kernel, KernelType
from TD.dataset import Dataset

class MailSelector:
    def __init__(self, train_file, test_file, kernel_type=KernelType.RBF, gamma=0.5, C=1.0, tol=1e-5):
        self.train_file = train_file
        self.test_file = test_file
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.C = C
        self.tol = tol

    def run(self):
        
        train_data = Dataset(self.train_file)
        test_data = Dataset(self.test_file)
        pass

