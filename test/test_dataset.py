import unittest
import numpy as np
from satsense.classification import Dataset

class DatasetTest(unittest.TestCase):    
    def setUp(self):
        tmp = np.arange(0, 24)
        # shape of feature vector is 2x3x4
        self.feature_vector = np.reshape(tmp, (2, 3, 4))
        self.mask = np.array([[0, 1, 0], [1, 0, 0]])
    
    def teardown(self):
        self.feature_vector = None
        self.mask = None
    
    def test_create_xy_x(self):
        dataset = Dataset(self.feature_vector)
        X,_ = dataset.createXY(self.mask)
        np.testing.assert_array_equal(X.shape, (2, 4))
        np.testing.assert_array_equal(X, np.array([[4,5,6,7], [12, 13, 14, 15]]))

    def test_create_xy_y(self):
        dataset = Dataset(self.feature_vector)
        _,y = dataset.createXY(self.mask)
        np.testing.assert_array_equal(y.shape, (2, ))
        np.testing.assert_array_equal(y, np.array([1, 1]))

    def test_create_xy_no_remove_out_x(self):
        dataset = Dataset(self.feature_vector)
        X,_ = dataset.createXY(self.mask, remove_out=False)

        np.testing.assert_array_equal(X.shape, (6, 4))
        np.testing.assert_array_equal(X, np.array([[0, 1, 2, 3,], [4, 5, 6, 7], [8, 9, 10, 11],
                                      [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]))
    
    def test_create_xy_no_remove_out_y(self):
        dataset = Dataset(self.feature_vector)
        _,y = dataset.createXY(self.mask, remove_out=False)
        np.testing.assert_array_equal(y.shape, (6, ))
        np.testing.assert_array_equal(y, np.array([0, 1, 0, 1, 0, 0]))

    def test_create_xy_in_out_label_swap_y(self):
        dataset = Dataset(self.feature_vector)
        _,y = dataset.createXY(self.mask, in_label=0, out_label=1, remove_out=False)
        np.testing.assert_array_equal(y.shape, (6, ))
        np.testing.assert_array_equal(y, np.array([1, 0, 1, 0, 1, 1]))
        dataset = None
        y = None

        dataset = Dataset(self.feature_vector)
        _,y = dataset.createXY(self.mask, in_label=0, out_label=1)
        np.testing.assert_array_equal(y.shape, (2, ))
        np.testing.assert_array_equal(y, np.array([0, 0]))

    def test_create_xy_no_remove_out_in_out_label_swap_y_(self):
        dataset = Dataset(self.feature_vector)
        _,y = dataset.createXY(self.mask, in_label=0, out_label=1)
        np.testing.assert_array_equal(y.shape, (2, ))
        np.testing.assert_array_equal(y, np.array([0, 0]))


if __name__ == "__main__":
    unittest.main()

        

