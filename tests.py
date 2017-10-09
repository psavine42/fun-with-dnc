import unittest
import dnc as dnc
import torch
import generators as gen
from torch.autograd import Variable
import numpy as np

sq_width = 6
sq_length = 4
bt_size = 1
n_heads = 1

class misc(unittest.TestCase):
    def setUp(self):

        self.data = gen.RandomData(seq_width=sq_length,
                                   seq_len=sq_width)
        self.Dnc = dnc.DNC(unit_size_W=sq_length,
                           batch_size=bt_size,
                           num_heads=n_heads)
        
    def partitions(self):
        prt = self.Dnc.interface_part()
        print(prt)
        flt = flat_list = [item for sublist in prt for item in sublist]
        print(flt, len(flt))
        assert len(flt) == self.Dnc.interface_size
    
    def dims(self):
        print(self.Dnc.controller.lst1.state_dict())

    def run_(self):
        itm, zer = self.data.__getitem__(0)
        hidn = (Variable(torch.zeros(2, bt_size, 250)),
                Variable(torch.zeros(2, bt_size, 250)))
        inpt = Variable(torch.from_numpy(itm).unsqueeze(1).float())
        print("input", inpt.size())
        print("hiddn", hidn[0].size())
        out1, (out2, out3) = self.Dnc(inpt, hidn)
        print("outs", out1.size(), out2.size(), out3.size())


class Setup(unittest.TestCase):
    def setUp(self):
        self.data = gen.RandomData(seq_width=sq_width,
                                   seq_len=sq_length)
    
    def get1(self):
        print(self.data.__getitem__(0))

if __name__ == "__main__":
    unittest.main()