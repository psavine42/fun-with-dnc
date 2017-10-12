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

"""
access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
      [Variable(torch.zeros(num_layers, batch_size, hidden_size)),
        Variable(torch.zeros(num_layers, batch_size, hidden_size))]
  }
"""
eps = 10e-6

class Controller(nn.Module):
    def __init__(self,
                 batch_size=32,
                 num_layers=2,
                 word_size=4,
                 output_size=24,
                 hidden_size=250):
        super(Controller, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_hidden = hidden_size
        #state
        print("controller {}, {}, {} -> out:{}".format(
            word_size, hidden_size, num_layers, output_size))
        self.lst1 = nn.LSTM(input_size=word_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False)
        self.map_to_output = nn.Linear(hidden_size, output_size)


    def forward(self, inputs, previous_controller_state):
        print(inputs.size(), self.lst1)
        lstm_inputs = inputs.permute(0, 2, 1)

        print(lstm_inputs.size(), )
        lstm_out, controller_state = self.lst1(lstm_inputs, previous_controller_state)
        print(lstm_out.size(), )
        
        outputs = self.map_to_output(lstm_out)
        return outputs, controller_state


class DNCMemory(nn.Module):
    def __init__(self,
                 word_size=32,
                 memory_size=100,
                 read_heads_h=1,
                 write_heads=1):
        self.W = word_size
        self.N = memory_size
        self.num_reads = read_heads_h
        self.num_writes = write_heads
        self._memory = Variable(torch.zeros(self.N , self.W).float())
        self._linkage = [] #addressing.TemporalLinkage(memory_size, num_writes)
        self._freeness = [] #addressing.Freeness(memory_size)

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

    def forward(self, inputs):
        #create
        inputs = self._read_inputs(inputs)

        # Update usage using inputs['free_gate'] and previous read & write weights
        # Write to memory
        #linkage_state = self._linkage(write_weights, prev_state.linkage)
        # Read from memory.
        read_words = tf.matmul(read_weights, self._memory)
        memory = []
        read_weights = []
        write_weights = []
        linkage_state = []
        usage = []
        return (read_words,
                (memory, read_weights, write_weights, linkage_state, usage))


class DNC(nn.Module):
    def __init__(self, 
                 batch_size=32,
                 #mem_rows__N=32,
                 memory_size=100,
                 word_len=6,
                 num_layers=2,
                 hidden_size=4,
                 num_inputs=250,
                 num_read_heads=1,
                 **kwdargs):
        super(DNC, self).__init__()

        self.word_len = word_len
        self.num_read_heads = num_read_heads
        self.interface_size = (self.num_read_heads * self.word_len) + \
                              (3 * self.word_len) + \
                              (5 * self.num_read_heads + 3)

        self._controller = Controller(word_size= word_len + num_read_heads,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      output_size=self.interface_size,
                                      batch_size=batch_size)

       # print()
        #print("DNC_memory w:{}, N:{}".format(self.Memory.W, self.Memory.N))
        #print("interface size: ", self.interface_weights.size())
        #print("unit_size", unit_size_W)
        #print("read vec", self.read_vector__r.size())
        #print()

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
        return (torch.index_select(xs, 0, i) for i in sizes)

    def forward(self,
                inputs,
                prev_access_output,
                prev_access_state,
                prev_controller_state):
        """
            `input`          [batch_size, seq_len, word_size]
            `access_output`
                `[batch_size, num_reads, word_size]` containing read words.
                    access output_size =
            `access_state` is a tuple of the access module's state
                memory,       [memory_size, word_size]
                read_weights,  W r [batch_size, num_reads, memory_size]
                write_weights, W [batch_size, num_writes, memory_size]
                linkage_state, L
                    links             [batch_size, num_writes, memory_size, memory_size]
                    precedence_wghts  [batch_size, num_writes, memory_size]
                usage,  F
            `controller_state` []
                :tuple of controller module's state

        """
        #prev_access_output, prev_access_state, prev_controller_state = previous_state
        print(type(inputs), type(prev_access_output))
        controller_input = torch.cat([inputs, prev_access_output], dim=1)

        controller_output, controller_state = self._controller(controller_input, prev_controller_state)
        print("controller out", controller_output.size())
        #access_output, access_state = self._Memory(controller_output, prev_access_state)
        access_output, access_state = [], []
        output = []
        #output = torch.cat((controller_output, access_output))
        return [output, access_output, access_state, controller_state]

def content_addressed_lookup(k):
    #F.cosine_similarity()
    pass

def start_state(num_layers=2,
                word_len=4,
                num_read_heads=1,
                num_write_heads=1,
                memory_size=100,
                batch_size=8,
                hidden_size=64):
    """ prev_state: A `DNCState` tuple containing the fields
        `access_output`,`access_state` and `controller_state`.
        `access_output` `[batch_size, num_reads, word_size]` containing read words.
        `access_state` is a tuple of the access module's state
        `controller_state` is a tuple of controller module's state
    """
    interface_size = num_read_heads * word_len + 3 * word_len +  5 * num_read_heads + 3
    #interface_size
    #memory state
    #controller state
    access_output = Variable(torch.zeros(batch_size, num_read_heads, word_len))
    memory = Variable(torch.zeros(word_len, memory_size))
    hidden = [Variable(torch.zeros(num_layers, batch_size, hidden_size)),
              Variable(torch.zeros(num_layers, batch_size, hidden_size))]
    return [access_output, memory, hidden]




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

if __name__ == "__main__":
    print("x")
