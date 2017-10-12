import unittest
import dnc as dnc
import torch
import generators as gen
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

sq_width = 4
sq_length = 6
bt_size = 1
n_heads = 1
num_layers=2, 
init_args= {'word_len':4,
            'num_layers':2,
            'num_read_heads':1,
            'num_write_heads':1,
            'memory_size':100,
            'batch_size':8, 
            'hidden_size':64}

class misc(unittest.TestCase):
    def setUp(self):

        self.data = gen.RandomData(seq_width=init_args['word_len'],
                                   seq_len=init_args['word_len'])
        self.loader = DataLoader(self.data, batch_size=init_args['batch_size'])
        self.Dnc = dnc.DNC(**init_args)
        
    def partitions(self):
        #prt = self.Dnc.interface_part()
        #print(prt)
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

        init_state = dnc.start_state(**init_args)

        out1, (out2, out3) = self.Dnc(inpt, hidn)
        print("outs", out1.size(), out2.size(), out3.size())

    def init_test(self):
        itm, zer = self.loader.__iter__().__next__()
        print("input", itm.size())
        initial_state = dnc.start_state(**init_args)
        access_output, access_state, hidden = initial_state
        
        print("access_output", access_output.size())
        print("access_state", access_state.size())

        print()
        out1, _, _, _ = self.Dnc(Variable(itm.float()), access_output, access_state, hidden )
        #print("hidden_out", h_o.size())
        #print("hidden_c", h_c.size())
    
    def controller_(self):
        dnc.Controller()
        


class Setup(unittest.TestCase):
    def setUp(self):
        self.data = gen.RandomData(seq_width=sq_width,
                                   seq_len=sq_length)
    
    def get1(self):
        print(self.data.__getitem__(0))

if __name__ == "__main__":
    unittest.main()


"""
num_seq = 10
    seq_len = 6
    seq_width = 4
    iterations = 1000
    con = np.random.randint(0, seq_width,size=seq_len)
    seq = np.zeros((seq_len, seq_width))
    seq[np.arange(seq_len), con] = 1
    end = np.asarray([[-1]*seq_width])
    zer = np.zeros((seq_len, seq_width))

    graph = tf.Graph()
    
    with graph.as_default():
        #training time
        with tf.Session() as sess:
            #init the DNC
            dnc = DNC(
                                                    batch_size    
            input_size=4, output_size=4, seq_len=6, num_words=10, word_size=4, num_heads=1)

                    word_
"""