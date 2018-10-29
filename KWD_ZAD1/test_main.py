import unittest
import functions
import pandas as pd

class TestMain(unittest.TestCase):

    @classmethod  
    def setUpClass(cls):
        print('setUpClass')

    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')


    def setUp(self):
        print('setUp')
        self.first = functions.KNN(3, pd.read_csv("iris.data.learning", header=None))
        self.second = functions.KNN(5, pd.read_csv("iris.data.learning", header=None))
        self.third = functions.KNN(7, pd.read_csv("iris.data.learning", header=None))


    def tearDown(self):
          print('tearDown')


    def testPredict(self):
        print('testing predict')
        res1 = self.first.predict(functions.getNumpyArrayFromCSV("iris.data.test")[:, 0:4], False)
        test_res1 = ['Iris-setosa', 'Iris-setosa', 
                        'Iris-setosa', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica']
        res2 = self.second.predict(functions.getNumpyArrayFromCSV("iris.data.test")[:, 0:4], False)
        test_res2 = ['Iris-setosa', 'Iris-setosa', 
                        'Iris-setosa', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica']
        res3 = self.third.predict(functions.getNumpyArrayFromCSV("iris.data.test")[:, 0:4], False)
        test_res3 = ['Iris-setosa', 'Iris-setosa', 
                        'Iris-setosa', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-versicolor', 
                        'Iris-versicolor', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica', 'Iris-virginica', 
                        'Iris-virginica']
        self.assertSequenceEqual(res1, test_res1)
        self.assertSequenceEqual(res2, test_res2)
        self.assertSequenceEqual(res3, test_res3)

    def testScore(self):
        print("testing score")
        score1 = self.first.score(functions.getNumpyArrayFromCSV("iris.data.test")[:, 0:4],functions.getNumpyArrayFromCSV("iris.data.test")[:, 4], False)
        score2 = self.second.score(functions.getNumpyArrayFromCSV("iris.data.test")[:, 0:4],functions.getNumpyArrayFromCSV("iris.data.test")[:, 4], False)
        score3 = self.third.score(functions.getNumpyArrayFromCSV("iris.data.test")[:, 0:4],functions.getNumpyArrayFromCSV("iris.data.test")[:, 4],False)

        self.assertEqual(93.33333333333333, score1)
        self.assertEqual(93.33333333333333, score2)
        self.assertEqual(93.33333333333333, score3)

    if __name__ == '__main__':
        unittest.main()