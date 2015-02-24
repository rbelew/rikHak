import unittest
import os

from json2djfix import json2djfix

TESTDATADIR = os.path.join(os.path.dirname(__file__))

class Test_json2djf(unittest.TestCase):

    def setUp(self):
        self.modelName,self.fldMap, self.pk = json2djfix.loadParams(TESTDATADIR+'/sffilm2djf.txt')

    def test_ParamLoad(self):
        self.assertEqual(self.modelName, 'SFFilm4LL')

        allKeys = self.fldMap.keys()
        allKeys.sort()

        sfKeys = ['actor_1', 'actor_2', 'actor_3', 'director', 'distributor', 'fun_facts', 'locations', 'production_company', 'release_year', 'title', 'writer']
        self.assertEqual(allKeys,sfKeys)

        allVals = [self.fldMap[k] for k in allKeys]
        sfVals = [None, None, None, 'director', None, None, 'location', None, 'year', 'title', 'writer']
        self.assertEqual(allVals,sfVals)

        self.assertEqual(self.pk, 'year')

if __name__ == '__main__':
    unittest.main()

