import unittest
import os
import FortranFile
from FortranFile import FortranFile as ff

class CorrectFile(unittest.TestCase):
    filename = 'correct_file'
    r1 = 'This is a test.'
    r2 = 'All the records in this file are correct.'
    r3 = 'No errors here.'

    def setUp(self):
        self.f = ff(self.filename)

    def tearDown(self):
        self.f.close()

    def testRead(self):
        r1 = self.f.readRecord()
        r2 = self.f.readRecord()
        r3 = self.f.readRecord()
        self.assertEqual(r1, self.r1)
        self.assertEqual(r2, self.r2)
        self.assertEqual(r3, self.r3)

    def testReadNoMoreRecords(self):
        self.f.readRecord()
        self.f.readRecord()
        self.f.readRecord()
        self.assertRaises(FortranFile.NoMoreRecords, self.f.readRecord)

    def testSkip(self):
        self.f.skipRecord()
        self.f.skipRecord()
        r3 = self.f.readRecord()
        self.assertEqual(r3, self.r3)

    def testSkipNoMoreRecords(self):
        self.f.skipRecord()
        self.f.skipRecord()
        self.f.skipRecord()
        self.assertRaises(FortranFile.NoMoreRecords, self.f.skipRecord)

    def testReadBack(self):
        self.f.seek(0,os.SEEK_END)
        r3 = self.f.readBackRecord()
        r2 = self.f.readBackRecord()
        r1 = self.f.readBackRecord()
        self.assertEqual(r3, self.r3)
        self.assertEqual(r2, self.r2)
        self.assertEqual(r1, self.r1)

    def testReadBackNoMoreRecords(self):
        self.assertRaises(FortranFile.NoMoreRecords, self.f.readBackRecord)

    def testSkipBack(self):
        self.f.seek(0,os.SEEK_END)
        self.f.skipBackRecord()
        self.f.skipBackRecord()
        r1 = self.f.readBackRecord()
        self.assertEqual(r1, self.r1)

    def testSkipBackNoMoreRecords(self):
        self.assertRaises(FortranFile.NoMoreRecords, self.f.skipBackRecord)

if __name__ == '__main__':
    unittest.main()
