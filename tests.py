import unittest
import dnc as dnc
import torch
import generators as gen
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torchvision.utils as vutils
import numpy as np
from tensorboardX import SummaryWriter


sq_width = 4
sq_length = 6
bt_size = 1
n_heads = 1
num_layers=2, 
init_args= {'word_len':4,
            'num_layers':2,
            'num_read_heads':2,
            'num_write_heads':1,
            'memory_size':100,
            'batch_size':5,
            'hidden_size':64}

class misc(unittest.TestCase):
    def setUp(self):
        self.writer = SummaryWriter()
        self.data = gen.RandomData(seq_width=init_args['word_len'],
                                   seq_len=init_args['num_read_heads'])
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
        itm, zer = self.loader.__iter__().__next__()
        inpt = Variable(itm.float())
        
        
        wt = Variable(torch.FloatTensor(init_args['batch_size'],
                                        1, init_args['word_len']).uniform_(0, 1))
        print("input", wt.size())
        init_state = dnc.start_state(**init_args)

        out1, out_state = self.Dnc(wt, init_state)
        #print("outs", out1.size(), out2.size(), out3.size())

    def init_test(self):
        itm, zer = self.loader.__iter__().__next__()
        print("input", itm.size())
        initial_state = dnc.start_state(**init_args)
        access_output, access_state, hidden = initial_state

        print("access_output", access_output.size())
        #print("access_state", access_state.size())

        print()
        out1, _, _, _ = self.Dnc(Variable(itm.float()), initial_state )
        #print("hidden_out", h_o.size())
        #print("hidden_c", h_c.size())

    def controller_(self):
        dnc.Controller()

    def freetest(self):
        _, initial_state, _ = dnc.start_state(**init_args)
        freenes = dnc.Usage(memory_size=100)
        batch_size = 3
        free_gate = Variable(torch.FloatTensor(batch_size, 1).uniform_(0, 1))
        mem, read_wghts, wrt_wghts, linkage, usage = initial_state

        wrt_wghts = torch.stack([wrt_wghts for i in range(batch_size)], 0)
        read_wghts = torch.stack([read_wghts for i in range(batch_size)], 0)
        usage = torch.stack([usage for i in range(batch_size)], 0)

        result = freenes(wrt_wghts, free_gate, read_wghts, usage)
        print(result.size())


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