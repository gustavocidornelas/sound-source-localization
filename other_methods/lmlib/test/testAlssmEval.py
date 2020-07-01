import unittest
from numpy.testing import *
from lmlib.statespace.model import Alssm


class AlssmTestCase(unittest.TestCase):
    def test_alssm_eval(self):
        # Single-Output ALSSM
        A = [[1, 1, 1], [0, 1, 2], [0, 0, 1]]
        C = [1, 0, 0]
        alssm = Alssm(A, C)
        x1 = [[1, 0, 0]]
        x2 = [[1, 0, 0], [2, 3, 4]]
        x3 = [[[1, 0.1], [0, 1], [0, 1]], [[2, 0], [3, 2], [4, 0.1]]]

        self.assertIsNone(assert_array_equal(alssm.eval(x1, time_range=0), [1]))
        self.assertIsNone(
            assert_array_equal(alssm.eval(x1, time_range=[-1, 0, 1]), [[1.0, 1.0, 1.0]])
        )
        self.assertIsNone(assert_array_equal(alssm.eval(x2, time_range=0), [1, 2]))
        self.assertIsNone(
            assert_array_equal(
                alssm.eval(x2, time_range=[-1, 0, 1]), [[1, 1, 1], [3, 2, 9]]
            )
        )
        self.assertIsNone(
            assert_array_equal(alssm.eval(x3, time_range=0), [[1, 0.1], [2, 0]])
        )
        self.assertIsNone(
            assert_array_almost_equal(
                alssm.eval(x3, time_range=[-1, 0, 1]),
                [[[1, 0.1], [1, 0.1], [1, 2.1]], [[3, -1.9], [2, 0], [9, 2.1]]],
            )
        )

        # Multi-Output ALSSM
        C = [[1, 0, 0]]
        alssm = Alssm(A, C)

        self.assertIsNone(assert_array_equal(alssm.eval(x1, time_range=0), [[1]]))
        self.assertIsNone(
            assert_array_equal(alssm.eval(x1, time_range=[-1, 0, 1]), [[[1], [1], [1]]])
        )
        self.assertIsNone(assert_array_equal(alssm.eval(x2, time_range=0), [[1], [2]]))
        self.assertIsNone(
            assert_array_equal(
                alssm.eval(x2, time_range=[-1, 0, 1]),
                [[[1], [1], [1]], [[3], [2], [9]]],
            )
        )
        self.assertIsNone(
            assert_array_equal(alssm.eval(x3, time_range=0), [[[1, 0.1]], [[2, 0]]])
        )
        self.assertIsNone(
            assert_array_almost_equal(
                alssm.eval(x3, time_range=[-1, 0, 1]),
                [
                    [[[1, 0.1]], [[1, 0.1]], [[1, 2.1]]],
                    [[[3, -1.9]], [[2, 0]], [[9, 2.1]]],
                ],
            )
        )


if __name__ == "__main__":
    unittest.main()
