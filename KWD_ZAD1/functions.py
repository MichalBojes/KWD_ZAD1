import numpy as np
import scipy.spatial.distance as sp
import pandas as pd
import test_main
import unittest


def getNumpyArrayFromCSV(nameOfFile):
    return np.array(pd.read_csv(nameOfFile, header=None))



class KNN:

    def __init__(self, k, trainList):
        self.k=k
        self.trainList=np.array(trainList)
        self.value = 0

    def predict(self, testListData, if_print=True):
        self.lista= list()
        self.distance= np.empty([len(testListData), len(self.trainList)], dtype=[('dist', float), ('label', 'U30')])
        for test in range (0, len(testListData), 1):
            if if_print:
                print("Punkt: %d" %test)
            for trained in range (0, len(self.trainList), 1):
                self.distance[test][trained]=sp.euclidean(testListData[test], self.trainList[trained, 0:4]), self.trainList[trained, 4]
            self.tosort=np.array(self.distance[test], dtype=[('dist', float), ('label', 'U30'),])
            self.tosort.sort(order='dist')
            self.thisdict = {}
            for i in range (self.k):
                self.tuple=self.tosort[i]
                if self.tuple[1] not in self.thisdict:
                    self.thisdict[self.tuple[1]]=1
                else:
                    self.thisdict[self.tuple[1]]+=1
            self.sorteddict=sorted(self.thisdict, key=self.thisdict.__getitem__, reverse=True)
            if if_print:
                print(self.sorteddict[0])
            self.lista.append(self.sorteddict[0])
            self.sorteddict.clear()
            self.thisdict.clear()
        return(self.lista)

    def score(self,test_list_data, predicted_labels_list, if_print=True):
        self.correctly=0
        self.labels=self.predict(test_list_data, False)
        for i in range (0,len(self.labels),1):
            if self.labels[i]==predicted_labels_list[i]:
                self.correctly+=1
        self.procentage=self.correctly/len(predicted_labels_list)*100
        if if_print:
            print("poprawnie rozpoznano (sztuk): %d na %d" % (self.correctly, len(predicted_labels_list)))
            print("poprawnie rozpoznano (w procentach): %d" %self.procentage )
        return self.procentage

 