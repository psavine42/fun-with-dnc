import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import generators as gen
from torch.utils.data import DataLoader

"""Controller from paper
    mini-shrdlu
    lstm -      2 x 250
    batch-size  32
    learning-rt 3e-5
    memory dims 32 x 100

    """

"""Combine input + read-head:
    """
"""Controller network. At every time-step t the controller network N receives an
    input vector xt∈RX from the dataset or environment and emits an output vector
    yt∈ RY that parameterizes either a predictive distribution for a target 
    vector zt∈ RY
    (supervised learning) or an action distribution (reinforcement learning).
    Additionally, the controller receives a set of R read vectors 
    r r t t − − , , … R
    from the
    memory matrix Mt−1∈ R^NxW at the previous time-step, via the read heads. It then
    emits an interface vector ξt that defines its interactions with the memory at the
    current time-step. For notational convenience, we concatenate the read and input
    vectors to obtain a single controller input vector χ = … [ ; x r − − ; ;r ] t t t tR 1 1 1 . Any
    neural network can be used for the controller, but we have used the following
    variant of the deep LSTM architecture
    """

class Controller(nn.Module):
    def __init__(self, 
                 in_unit_size=6, 
                 batch_size=32, 
                 layers=2, 
                 size=250):
        super(Controller, self).__init__()
        self.num_layers = layers
        self.batch_size = batch_size
        self.num_hidden = size
        print(in_unit_size, size, layers)
        self.lst1 = nn.LSTM(input_size=in_unit_size, 
                            hidden_size=size, 
                            num_layers=layers,
                            bidirectional=False)
    def init_hidden(self):
        return Variable(weight.new(self.num_layers, self.batch_size, self.num_hidden).zero_())

    def forward(self, x, hidden):
        return self.lst1(x, hidden)
        

class DNCMemory():
    def __init__(self, W=100, N=32, R_h=1, W_h=1):
        self.W = W
        self.N = N
        self.memory = Variable(torch.zeros(N, W).float())
    
    def content_lookup(self, key, weights):
        norm_mem = torch.norm(self.memory, p=2, dim=1).detach()
        norm_key = torch.norm(key, 0)
        sim = torch.dot(norm_mem, norm_key) 
        #str is 1*1 or 1*R
        #returns similarity measure
        return F.softmax(sim * weights, 0)

    def Write(self, w_t_W, e_t, v_t):
        "Reading and writing to memory"
        ones = torch.ones(self.N, self.W)
        erase = ones - torch.dot(w_t_W, e_t)
        M = torch.mul(self.memory, erase) + torch.dot(w_t_W, v_t)
        return M


class DNC(nn.Module):
    def __init__(self, batch_size=32,
                 N=32,
                 unit_size_W=6, 
                 num_heads=1):
        super(DNC, self).__init__()
        self.unit_size_W = unit_size_W
        self.num_read_heads = num_heads
        self.dnc_rows_N = N
        #compute interface size
        self.interface_size = (self.num_read_heads * self.unit_size_W) + \
                              (3 * self.unit_size_W) + \
                              (5 * self.num_read_heads + 3)
        self.Memory = DNCMemory(W=unit_size_W, N=N)
        self.nn_output_size = self.interface_size + unit_size_W
        self.interface_weights = \
            Variable(torch.zeros(self.nn_output_size, self.interface_size))
        self.controller = Controller(in_unit_size=unit_size_W,
                                     batch_size=batch_size)
    
    def interface_part(self):
        partition = [[0] * (self.num_read_heads * self.unit_size_W),
                     [1] * (self.num_read_heads),
                     [2] * (self.unit_size_W), [3],
                     [4] * (self.unit_size_W),
                     [5] * (self.unit_size_W),
                     [6] * (self.num_read_heads), 
                     [7], 
                     [8],
                     [9] * (self.num_read_heads * 3)]
        ds = []
        cntr = 0
        for idx in partition:
            if len(idx) == 1:
                ds.append(torch.tensor([cntr]))
            else:
                ds.append(torch.tensor([cntr, cntr + len(idx) - 1]))
            cntr += len(idx)
        return ds
    
    def partition_components(self, xs):
        sizes = self.interface_part()
        return (torch.index_select(xs, dim=0, i) for i in sizes)

    def forward(self, x, hidden):
        #convert interface vector into a set of read write vectors
        #run controller forward
        out_v_T, out_Eta = self.controller(x, hidden)
        
        #torch.index_select(out_v_T, 1, [0, 2])
        #interface_vec_eta = tf.matmul(l2_act, self.interface_weights)
        y_T = out_v_T 
        """
        read_keys_k_t = 
        read_weights = 
        write_keys_k_t =
        write_str_B_t = 
        erase_vec_e_t = 
        write_vec_v_t = 
        free_gates_f_t =
        allocation_gate_g_t =
        write_gate_g_t = 
        read_modes_R = 
        """
        return out_v_T, out_Eta

def content_addressed_lookup(k_, ):
    F.cosine_similarity()
    pass

def one_hot(x):
    pass

"""w_t -> write weights, e_t"""
def oneplus(x):
    return 1 + torch.log(1 + e ** x)

def controller_output(input_x, input_R):
    return input_x + input_R

def run_dnc(batch_size=32):
    data_gen = gen.RandomData()
    loader = DataLoader(data_gen, batch_size=batch_size)
    dnc = DNC(batch_size=32)
    for i, (trgt, label) in enumerate(loader):
        #output = dnc(trgt)
        pass

if __name__ == "__main__":
    print("x")
