import numpy as np
import scipy.spatial.distance as sp
import pandas as pd
import functions
import unittest




kNN = functions.KNN(3, pd.read_csv("iris.data.learning", header=None))
kNN.predict(functions.getNumpyArrayFromCSV("iris.data.test")[:, 0:4])
kNN.score(functions.getNumpyArrayFromCSV("iris.data.test")[:, 0:4],functions.getNumpyArrayFromCSV("iris.data.test")[:, 4])

suite = unittest.TestLoader().loadTestsFromModule(functions.test_main)
unittest.TextTestRunner(verbosity=2).run(suite)



