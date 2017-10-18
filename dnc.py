import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import generators as gen
from torch.utils.data import DataLoader
from utils import *

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
init_args= {'word_len':4,
            'num_layers':2,
            'num_reads':2,
            'num_writes':1,
            'memory_size':100,
            'batch_size':5,
            'hidden_size':64}
eps = 10e-6


class Controller(nn.Module):
    def __init__(self,
                 batch_size=32,
                 num_reads=2,
                 num_layers=2,
                 word_size=4,
                 output_size=24,
                 hidden_size=250):
        super(Controller, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_hidden = hidden_size
        self._insize = word_size + word_size * num_reads
        #state
        print("controller {}, {}, {} -> out:{}".format(
            word_size, hidden_size, num_layers, output_size))
        self.lst1 = nn.LSTM(input_size=self._insize, #word_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.lst2 = nn.LSTM(input_size=hidden_size, #word_size,
                            hidden_size=output_size,
                            num_layers=1,
                            batch_first=True)
        self.map_to_output = nn.Linear(hidden_size, output_size)

        #self.interface_weights = Variable(torch.FloatTensor(output_size - word_size, batch_size).uniform_(0, 1))
        #self.read_weights = Variable(torch.FloatTensor(word_size, batch_size).uniform_(0, 1))

    def forward(self, inputs, previous_controller_state):
        """Generate interface and output vector
            """
        print()
        inputs = inputs.view(-1, self._insize).unsqueeze(1)
        print("lstm input", inputs.size(), self.lst1)
        lstm_out, controller_state = self.lst1(inputs, previous_controller_state)
        print("lstm out", lstm_out.size())

        outputs = self.map_to_output(lstm_out).squeeze()
        print("map out", outputs.size())
        #read_outputs = outputs.matmul(self.read_weights)
        #interface = outputs.matmul(self.interface_weights)

        #print("read_outputs", read_outputs.size())
        #print("interface", interface.size())
        return outputs, controller_state
        #return [read_outputs, interface], controller_state


class Usage(nn.Module):
    def __init__(self, memory_size=100):
        super(Usage, self).__init__()
        self._memory_size = memory_size

    def _usage_after_write(self, prev_usage, write_weights):
        """Calcualtes the new usage after writing to memory.

            Args:
            prev_usage: tensor of shape `[batch_size, memory_size]`.
            write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.

            Returns:
            New usage, a tensor of shape `[batch_size, memory_size]`.
            """
        # Calculate the aggregated effect of all write heads
        write_weights = 1 - (1 - write_weights).prod(1)
        return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        """Calcualtes the new usage after reading and freeing from memory.

            Args:
            prev_usage: tensor of shape `[batch_size, memory_size]`.

            free_gate: tensor of shape `[batch_size, num_reads]` 

                with entries in therange [0, 1] indicating the amount that locations 
                read from can be freed.
            read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.

            Returns:
            New usage, a tensor of shape `[batch_size, memory_size]`.
            """
        print()
        print("_usage_after_read")
        print("prev_usage  ", prev_usage.size())
        print("free_gate   ", free_gate.size())
        print("read_weights", read_weights.size())
        assert prev_usage.size(0) == init_args['batch_size']
        assert prev_usage.size(1) == init_args['memory_size']
        assert free_gate.size(0) == init_args['batch_size']
        assert free_gate.size(1) == init_args['num_reads']
        assert read_weights.size(0) == init_args['batch_size']
        assert read_weights.size(1) == init_args['num_reads']
        assert read_weights.size(2) == init_args['memory_size']

        free_gate = free_gate.unsqueeze(-1)
        free_read_weights = 1 - free_gate * read_weights
        phi = free_read_weights.prod(1)
        return prev_usage * phi

    def forward(self, write_weights, free_gate, read_weights, prev_usage):
        """
            Args:
            write_weights: tensor of shape `[batch_size, num_writes, memory_size]`
                giving write weights at previous time step.
            free_gate: tensor of shape `[batch_size, num_reads]` which indicates
                which read heads read memory that can now be freed.
            read_weights: tensor of shape `[batch_size, num_reads, memory_size]`
                giving read weights at previous time step.
            prev_usage: tensor of shape `[batch_size, memory_size]` giving
                usage u_{t - 1} at the previous time step, with entries in range
                [0, 1].

            Return tensor of shape `[batch_size, memory_size]`
            """
        write_weights = write_weights.detach()
        usage = self._usage_after_write(prev_usage, write_weights)
        print("usage1", usage.size())
        usage = self._usage_after_read(usage, free_gate, read_weights)
        print("usage2", usage.size())
        return usage

class Linkage(nn.Module):
    def __init__(self, memory_size=100, num_writes=1):
        super(Linkage, self).__init__()
        self._memory_size = memory_size
        self._num_writes = num_writes # todo

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        """Calculates the new precedence weights given the current write weights.

            The precedence weights are the "aggregated write weights" for each write
            head, where write weights with sum close to zero will leave the precedence
            weights unchanged, but with sum close to one will replace the precedence
            weights.

            Args:
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` containing the previous precedence weights.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new write weights.

            Returns:
            A tensor of shape `[batch_size, num_writes, memory_size]` containing the
            new precedence weights.
            """
        write_sum = write_weights.sum(2, keepdim=True)
        return (1 - write_sum) * prev_precedence_weights + write_weights

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """Calculates the new link graphs.

            For each write head, the link is a directed graph (represented by a matrix
            with entries in range [0, 1]) whose vertices are the memory locations, and
            an edge indicates temporal ordering of writes.

            Args:
                prev_link:
                    A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
                    representing the previous link graphs for each write head.
                prev_precedence_weights:
                    A tensor of shape `[batch_size, num_writes, memory_size]`
                    which is the previous "aggregated" write weights for each write head.
                write_weights:
                    A tensor of shape `[batch_size, num_writes, memory_size]`
                    containing the new locations in memory written to.

            Returns:
                A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
                containing the new link graphs for each write head.
            """

        #batch_size = prev_link.size(0)
        write_weights_i = write_weights.unsqueeze(3)
        write_weights_j = write_weights.unsqueeze(2)

        prev_precedence_weights_j = prev_precedence_weights.unsqueeze(2)
        prev_link_scale = 1 - write_weights_i - write_weights_j
        new_link = write_weights_i * prev_precedence_weights_j
        #scale old links, and add new links
        link = prev_link_scale * prev_link + new_link
            # Return the link with the diagonal set to zero, to remove self-looping edges.
            #  b w i j1 ...jn
            # [ [ [ [ 0 1 0 ]
            #       [ 1 0 0 ]
            #       [ 0 0 0 ]]]]
            #tf.matrix_set_diag(
                #link,
                #tf.zeros(
                    #[batch_size, self._num_writes, self._memory_size],
                    #dtype=link.dtype))
        return link

    def forward(self, write_weights, prev_state):
        """Calculate the updated linkage state given the write weights.

            Args:
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the memory addresses of the different write heads.
            prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
                shape `[batch_size, num_writes, memory_size, memory_size]`, and a
                tensor `precedence_weights` of shape `[batch_size, num_writes,
                memory_size]` containing the aggregated history of recent writes.

            Returns:
            A `TemporalLinkageState` tuple `next_state`, which contains the updated
            link and precedence weights.
            """
        prev_link, prev_precedence_weights = prev_state
        link = self._link(prev_link, prev_precedence_weights, write_weights)
        precedence_weights = self._precedence_weights(prev_precedence_weights, write_weights)
        return link, precedence_weights

class BatchSoftmax(nn.Module):
    def forward(self, input_):
        batch_size = input_.size(0)
        output_ = torch.stack([F.softmax(input_[i]) for i in range(batch_size)], 0)
        return output_


class WeightFn(nn.Module):
    def __init__(self, word_size=32, num_heads=1):
        super(WeightFn, self).__init__()
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = nn.Softplus()
        self.softmax = BatchSoftmax()

    def _vector_norms(self, m):
        mem_squared = m * m
        squared_norms = mem_squared.sum(2, keepdim=True) + eps
        return squared_norms.sqrt()

    def forward(self, inputs):
        """Connects the CosineWeights module into the graph.

            Args:
            memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
            keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
            strengths: A 2-D tensor of shape `[batch_size, num_heads]`.

            Returns:
            Weights tensor of shape `[batch_size, num_heads, memory_size]`.
            """
        memory, keys, strengths = inputs
        print()
        # Calculates the inner product between the query vector and words in memory.
        print(memory.size(), keys.size(), strengths.size())
        dot = keys.matmul(memory.transpose(1, 2))

        # Outer product to compute denominator (euclidean norm of query and memory).
        memory_norms = self._vector_norms(memory)
        key_norms = self._vector_norms(keys)

        print(key_norms.size(), memory_norms.size())
        norm = key_norms.matmul(memory_norms.transpose(1, 2))

        # Calculates cosine similarity between the query vector and words in memory.
        print("dot", dot.size(), norm.size())
        activations = dot / (norm + eps)

        #Weighted softmax as in paper.
        transformed_strengths = self._strength_op(strengths) #.unsqueeze(-1)

        sharp_activations = activations * transformed_strengths
        print("activations", transformed_strengths.size())
        print("transformed_strengths", transformed_strengths.size())
        print("sharp_activations", sharp_activations.size())
        print()
        return self.softmax(sharp_activations)


class DNCMemory(nn.Module):
    def __init__(self, word_size=32, memory_size=100, num_reads=1, write_heads=1):
        super(DNCMemory, self).__init__()
        self.W = word_size
        self.N = memory_size
        self.num_reads = num_reads
        self.num_writes = write_heads

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

        self._read_content_wght = WeightFn(word_size=word_size, num_heads=num_reads)
        self._write_content_wght = WeightFn(word_size=word_size, num_heads=write_heads)

        self._linkage = Linkage(memory_size=memory_size, num_writes=write_heads)
        self._usage = Usage(memory_size)
        self.parts = interface_part(num_reads, word_size)
        print(self.parts)

    def _allocation(self, usage):
        """Computes allocation by sorting `usage`

            This corresponds to the value a = a_t[phi_t[j]] in the paper.

            Args:
                usage: tensor of shape `[batch_size, memory_size]` indicating current
                    memory usage. This is equal to u_t in the paper when we only have one
                    write head, but for multiple write heads, one should update the usage
                    while iterating through the write heads to take into account the
                    allocation returned by this function.

            Returns:
                Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
            """
        # usage => [batch_size, memory_size]
        # Ensure values are not too small prior to cumprod.
        usage = eps + (1 - eps) * usage

        nonusage = 1 - usage
        #sorted_nonusage, indices = nonusage.sort(-1)
        #sorted_usage = 1 - sorted_nonusage
        #prod_sorted_usage = sorted_usage.cumprod(1)
        #
        #sorted_allocation = sorted_nonusage * prod_sorted_usage

        #inverse_indices = util.batch_invert_permutation(indices)
        #unpacked = permutations.unstack()
        #inverses = [tf.invert_permutation(permutation) for permutation in unpacked]
        #inverse_indices = tf.stack(inverses)
        # This final line "unsorts" sorted_allocation, so that the indexing
        # corresponds to the original indexing of `usage`.
        #return util.batch_gather(sorted_allocation, inverse_indices)
        return nonusage


    def write_allocation_weights(self, usage, write_gates):
        """Calculates freeness-based locations for writing to

            This finds unused memory by ranking the memory locations by usage, for each
            write head. (For more than one write head, we use a "simulated new usage"
            which takes into account the fact that the previous write head will increase
            the usage in that area of the memory.)

            Args:
                usage: A tensor of shape `[batch_size, memory_size]` representing
                    current memory usage.
                write_gates: A tensor of shape `[batch_size, num_writes]` with values in
                    the range [0, 1] indicating how much each write head does writing
                    based on the address returned here (and hence how much usage
                    increases).
                num_writes: The number of write heads to calculate write weights for.

            Returns:
                tensor of shape `[batch_size, num_writes, memory_size]` containing the
                    freeness-based write locations. Note that this isn't scaled by
                    `write_gate`; this scaling must be applied externally.
            """

        # expand gatings over memory locations
        write_gates = write_gates.unsqueeze(-1)

        allocation_weights = []
        for i in range(self.num_writes):
            #########################################################
            allocation_weights.append(self._allocation(usage))
            #########################################################
            # update usage to take into account writing to this new allocation
            usage += ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])

        # Pack the allocation weights for the write heads into one tensor.
        return torch.stack(allocation_weights, dim=1)


    def _write_weights(self, inputs, memory, usage):
        """Calculates the memory locations to write to.

            This uses a combination of content-based lookup and finding an unused
            location in memory, for each write head.

            Args:
            inputs: Collection of inputs to the access module, including controls for
                how to chose memory writing, such as the content to look-up and the
                weighting between content-based and allocation-based addressing.
            memory: A tensor of shape  `[batch_size, memory_size, word_size]`
                containing the current memory contents.
            usage: Current memory usage, which is a tensor of shape `[batch_size,
                memory_size]`, used for allocation-based addressing.

            Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` indicating where
                to write to (if anywhere) for each write head.
            """
        print()
        print("write_weights")
        alloc_gate, write_str, write_key, write_gate = inputs

        # c_t^{w, i} - The content-based weights for each write head.
        write_contnt_wghts = self._write_content_wght([memory, write_key, write_str])

        # a_t^i - The allocation weights for each write head.
        #write_alloc_wghts = self.write_allocation_weights(usage, (alloc_gate * write_gate))
        print(usage.size())

        write_alloc_wghts = self._allocation(usage)
        print("alloc", alloc_gate.size())
        print("alloc_wghts", write_alloc_wghts.size())
        print("alloc", write_contnt_wghts.size())
        # Expands gates over memory locations.
        #alloc_gate = alloc_gate.unsqueeze(-1)
        #write_gate = write_gate.unsqueeze(-1)

        # w_t^{w, i} - The write weightings for each write head.
        return write_gate * (alloc_gate * write_alloc_wghts + (1 - alloc_gate) * write_contnt_wghts)


    def directional_read_weights(self, link, prev_read_weights):
        """Calculates the forward or the backward read weights.

                For each read head (at a given address), there are `num_writes` link graphs
                to follow. Thus this function computes a read address for each of the
                `num_reads * num_writes` pairs of read and write heads.

                Args:
                link:
                    tensor-shape `[batch_size, num_writes, memory_size, memory_size]`
                    representing the link graphs L_t.
                prev_read_weights:
                    tensor-shape `[batch_size, num_reads, memory_size]`
                    containing the previous read weights w_{t-1}^r.
                forward: Boolean indicating whether to follow the "future" direction in
                    the link graph (True) or the "past" direction (False).

                Returns:
                tensor of shape `[batch_size, num_reads, num_writes, memory_size]`
                """
        # We calculate the forward and backward directions for each pair of
        # read and write heads; hence we need to tile the read weights and do a
        # sort of "outer product" to get this.
        expanded_read_weights = torch.stack([prev_read_weights] * self.num_writes, 1)
        result = expanded_read_weights.matmul(link)
        # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
        return result.permute(0, 2, 1, 3)

    def _read_weights(self, inputs, memory, prev_read_weights, link):
        """Calculates read weights for each read head.

            The read weights are a combination of following the link graphs in the
            forward or backward directions from the previous read position, and doing
            content-based lookup. The interpolation between these different modes is
            done by `inputs['read_mode']`.

            Args:
            inputs: Controls for this access module. This contains the content-based
                keys to lookup, and the weightings for the different read modes.
            memory: A tensor of shape `[batch_size, memory_size, word_size]`
                containing the current memory contents to do content-based lookup.
            prev_read_weights: A tensor of shape `[batch_size, num_reads,
                memory_size]` containing the previous read locations.
            link: A tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` containing the temporal write transition graphs.

            Returns:
            A tensor of shape `[batch_size, num_reads, memory_size]` containing the
            read weights for each read head.
            """
        #with tf.name_scope( 'read_weights', values=[inputs, memory, prev_read_weights, link]):
        read_keys, read_str, read_modes = inputs

        # c_t^{r, i} - The content weightings for each read head.
        content_weights = self._read_content_wght([memory, read_keys, read_str])

        # Calculates f_t^i and b_t^i.
        forward_weights = self.directional_read_weights(link, prev_read_weights.transpose(2, 3))
        backward_weights = self.directional_read_weights(link, prev_read_weights)

        backward_mode = read_modes[:, :, :self.num_writes]
        forward_mode = read_modes[:, :, self.num_writes:2 * self.num_writes]
        content_mode = read_modes[:, :, 2 * self.num_writes]

        content_str = content_mode.unsqueeze(2) * content_weights
        forward_str = (forward_mode.unsqueeze(3) * forward_weights).sum(2)
        backwrd_str = (backward_mode.unsqueeze(3) * backward_weights).sum(2)
        return content_str + forward_str + backwrd_str

    def _read_inputs(self, inputs):
        pass

    def interface_chunk(self, inputs, dim1, dim2, activation=None):

        num = dim2 * dim1
        #print(dim1, dim2)
        output = inputs[:, 0:num]
        rest = inputs[:, num:]
        #print(output)
        output = output.contiguous().view(-1, dim1, dim2)
        return output, rest

    def forward(self, inputs, prev_state):
        """ forward
            """
        print()
        print("access")

        [read_keys, read_str, write_key, write_str,
        erase_vec, write_vec, free_gates, alloc_gate,
                write_gate, read_modes] = inputs
        prev_memory, prev_read_wghts, prev_write_weights, prev_linkage, prev_usage = prev_state


        #print(self.parts)

        read_str = 1 + self.softplus(read_str)
        erase_vec = self.sigmoid(erase_vec)
        free_gates = self.sigmoid(free_gates.squeeze(-1))
        alloc_gate = self.sigmoid(alloc_gate)
        write_gate = self.sigmoid(write_gate)
        print("read_modes", read_modes.size())
        print("free_gates", free_gates.size())
        # 1 Update usage using inputs['free_gate'] and previous read & write weights.
        usage = usage = self._usage(prev_write_weights, free_gates, prev_read_wghts, prev_usage)
        print("usage", usage.size())
        # 2 Get Write weights
        write_inputs = [alloc_gate, write_str, write_key, write_gate]
        write_weights = self._write_weights(write_inputs, prev_memory, usage)
        print("write_wghts", write_weights.size())
        # 3 Write to memory.
        memory = erase_and_write(prev_memory, write_weights, erase_vec, write_vec)

        # 4 update linkage
        linkage_state = self._linkage(write_weights, prev_linkage)
        links, _ = linkage_state

        # 5 read from memory
        read_inputs = [read_keys, read_str, read_modes]
        read_weights = self._read_weights(read_inputs, memory, prev_read_wghts, links)

        # 6 read memory
        read_words = read_weights.matmul(memory)

        return (read_words,
                (memory, read_weights, write_weights, linkage_state, usage))


class InterFace(nn.Module):
    def __init__(self, word_size=32, num_reads=1, write_heads=1):
        super(InterFace, self).__init__()
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = write_heads
        self.batch_softmax = BatchSoftmax()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        #self.

    def _chunk1(self, inputs, dim1, activation=None):
        #print(dim1, dim2)
        output = inputs[:, :dim1]
        rest = inputs[:, dim1:]
        #output = output.contiguous().view(-1, dim1, dim2)
        if activation is not None:
            output = activation(output)
        return output, rest

    def _chunk2(self, inputs, dim1, dim2, activation=None):
        num = dim2 * dim1
        #print(dim1, dim2)
        output = inputs[:, 0:num]
        rest = inputs[:, num:]
        output = output.contiguous().view(-1, dim1, dim2)
        if activation is not None:
            output = activation(output)
        return output, rest

    def forward(self, inputs):
        read_keys, rest = self._chunk2(inputs, self.num_reads, self.word_size)
        read_str, rest = self._chunk1(rest, self.num_reads)

        write_key, rest = self._chunk2(rest, self.num_writes, self.word_size)
        write_str, rest = self._chunk1(rest, self.num_writes)

        erase_vec, rest = self._chunk2(rest, self.num_writes, self.word_size, self.sigmoid)
        write_vec, rest = self._chunk2(rest, self.num_writes, self.word_size)

        free_gates, rest = self._chunk1(rest, self.num_reads, self.sigmoid)
        alloc_gate, rest = self._chunk1(rest, self.num_writes, self.sigmoid)
        write_gate, rest = self._chunk1(rest, self.num_writes, self.sigmoid)

        read_modes, _ = self._chunk2(rest, self.num_reads, 3, self.batch_softmax)

        return [read_keys, read_str, write_key, write_str,
                erase_vec, write_vec, free_gates, alloc_gate,
                write_gate, read_modes]





class DNC(nn.Module):
    def __init__(self, batch_size=32, memory_size=100, word_len=6,
                 num_layers=2, hidden_size=4, num_read_heads=1, **kwdargs):
        super(DNC, self).__init__()

        self.word_len = word_len
        self.num_read_heads = num_read_heads
        self.interface_size = num_read_heads * word_len + 3 * word_len +  5 * num_read_heads + 3
        self._interface = InterFace(word_size=word_len, num_reads=num_read_heads, write_heads=1)
        self._controller = Controller(word_size=word_len,
                                      num_reads=num_read_heads,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      output_size=self.interface_size + word_len,
                                      batch_size=batch_size)
        self._Memory = DNCMemory(word_size=word_len, memory_size=memory_size, num_reads=num_read_heads)


    def forward(self, inputs, previous_state):
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
        prev_access_output, prev_access_state, prev_controller_state = previous_state
        print(inputs.size(), prev_access_output.size())
        print("interface vec", self.interface_size)
        controller_input = torch.cat([inputs, prev_access_output], 1)

        controller_output, controller_state = self._controller(controller_input, prev_controller_state)
        interface = self._interface(controller_output)
        #read_weights, interface = controller_output
        #print("controller out", read_weights.size(), interface.size())
        access_output, access_state = self._Memory(interface, prev_access_state)

        print("access_out", access_output.size())
        output = torch.cat([controller_output, access_output], 1)
        #output = torch.cat((controller_output, access_output))
        return [output, access_output, access_state, controller_state]




def content_addressed_lookup(k):
    #F.cosine_similarity()
    pass

def start_state(num_layers=2, word_len=4, num_read_heads=1, memory_size=100,
                batch_size=8, hidden_size=64, **kwdargs):
    """ prev_state: A `DNCState` tuple containing the fields
        `access_output` `[batch_size, num_reads, word_size]` containing read words.
        `access_state` is a tuple of the access module's state
        `controller_state` is a tuple of controller module's state
        """
    num_writes = 1

    access_output = Variable(torch.zeros(batch_size, num_read_heads, word_len))
    memory = [Variable(torch.zeros(batch_size, memory_size, word_len)), #memory
              Variable(torch.zeros(batch_size, num_read_heads, memory_size)), #read_weights
              Variable(torch.zeros(batch_size, num_writes, memory_size)), #write weights
              [Variable(torch.zeros(batch_size, num_writes, memory_size, memory_size)),
               Variable(torch.zeros(batch_size, num_writes, memory_size))], #linkage
              Variable(torch.ones(batch_size, memory_size)) #usage
             ]
    hidden = [Variable(torch.zeros(num_layers, batch_size, hidden_size)),
              Variable(torch.zeros(num_layers, batch_size, hidden_size))]
    return [access_output, memory, hidden]




def one_hot(x):
    pass

"""w_t -> write weights, e_t"""


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
