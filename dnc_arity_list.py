import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
from utils import _variable
# import logger as sl
from utils import repackage

eps = 10e-6


class VanillaLSTM(nn.Module):
    def __init__(self, batch_size=32,
                 num_reads=2,
                 num_layers=2,
                 input_size=None,
                 output_size=24,
                 hidden_size=250):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.out_size = output_size
        self.num_reads = num_reads

        self.lst1 = nn.LSTM(input_size=input_size + output_size * num_reads,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.lstout = nn.LSTM(input_size=hidden_size,
                              hidden_size=output_size,
                              num_layers=1,
                              batch_first=True)

    def forward(self, inputs, previous_state):
        """Generate interface and output vector"""
        prev_out, (ho1, hc1), (hc2, hc2) = previous_state

        control_input = torch.cat([inputs, prev_out], 1).unsqueeze(1)
        # print(control_input.size())
        lstm1_out, hidden1_out = self.lst1(control_input, (ho1, hc1))
        outputs, hidden2_out = self.lstout(lstm1_out, (hc2, hc2))

        final_out = outputs.squeeze(0)
        mock_access = torch.cat([final_out, prev_out[:, :self.out_size]], 1)
        # control_out = torch.cat([outputs, ], 1)
        return final_out, (mock_access, hidden1_out, hidden2_out)

    def init_state(self):
        return [_variable(torch.zeros(self.batch_size, self.num_reads * self.out_size), requires_grad=True),
                [_variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size), requires_grad=True),
                 _variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size), requires_grad=True)],
                [_variable(torch.randn(1, self.batch_size, self.out_size), requires_grad=True),
                 _variable(torch.randn(1, self.batch_size, self.out_size), requires_grad=True)]]


class Controller(nn.Module):
    def __init__(self,
                 batch_size=32,
                 num_reads=2,
                 num_layers=2,
                 input_size=None,
                 word_size=4,
                 output_size=24,
                 hidden_size=250):
        super(Controller, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_hidden = hidden_size
        if input_size is not None:
            self._insize = input_size
        else:
            self._insize = word_size + word_size * num_reads
        self.lst1 = nn.LSTM(input_size=self._insize,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.lst2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=output_size,
                            num_layers=1,
                            batch_first=True)
        self.map_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, previous_controller_state):
        """Generate interface and output vector"""
        inputs2 = inputs.unsqueeze(1)
        # print(inputs2.size())
        lstm_out, controller_state = self.lst1(inputs2, previous_controller_state)

        outputs = self.map_to_output(lstm_out).squeeze(1)
        return outputs, controller_state


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
        write_wghts = repackage(write_weights)
        usage = self._usage_after_write(prev_usage, write_wghts)
        usage = self._usage_after_read(usage, free_gate, read_weights)
        return usage


class Linkage(nn.Module):
    def __init__(self, memory_size=100, num_writes=1):
        super(Linkage, self).__init__()
        self._memory_size = memory_size
        self._num_writes = num_writes # todo

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

        # batch_size = prev_link.size(0)
        write_weights_i = write_weights.unsqueeze(3)
        write_weights_j = write_weights.unsqueeze(2)

        prev_precedence_weights_j = prev_precedence_weights.unsqueeze(2)
        prev_link_scale = 1 - write_weights_i - write_weights_j
        new_link = write_weights_i * prev_precedence_weights_j

        # scale old links, and add new links
        link = prev_link_scale * prev_link + new_link
        zeros = torch.LongTensor(list(range(self._memory_size)))
        zero_idxs = _variable(zeros.view(1, self._num_writes, -1, 1))
        return link.scatter_(-1, zero_idxs, 0)


    def forward(self, write_weights, prev_link, precedence_weights):
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
        assert list(write_weights.size()[1:]) == [1, self._memory_size]

        link = self._link(prev_link, precedence_weights, write_weights)
        # Calculates the new precedence weights given the current write weights.
        #   The precedence weights are the "aggregated write weights" for each write
        #    head, where write weights with sum close to zero will leave the precedence
        #    weights unchanged, but with sum close to one will replace the precedence weights.
        write_sum = write_weights.sum(2, keepdim=True)
        precedence_weights = (1 - write_sum) * precedence_weights + write_weights

        assert list(link.size()[1:]) == [1, self._memory_size, self._memory_size]
        assert list(precedence_weights.size()[1:]) == [1, self._memory_size]

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
        """CosineWeights

            Args:
            memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
            keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
            strengths: A 2-D tensor of shape `[batch_size, num_heads]`.

            Returns:
            Weights tensor of shape `[batch_size, num_heads, memory_size]`.
            """
        memory, keys, strengths = inputs
        assert list(memory.size())[2] == self._word_size
        assert list(keys.size())[1:] == [self._num_heads, self._word_size]
        assert list(strengths.size())[1:] == [self._num_heads]

        # Calculates the inner product between the query vector and words in memory.
        dot = keys.bmm(memory.transpose(1, 2))

        # Outer product to compute denominator (euclidean norm of query and memory).
        memory_norms = self._vector_norms(memory)
        key_norms = self._vector_norms(keys)
        norm = key_norms.bmm(memory_norms.transpose(1, 2))

        # Calculates cosine similarity between the query vector and words in memory.
        activations = dot / (norm + eps)
        transformed_strengths = self._strength_op(strengths).unsqueeze(-1)

        sharp_activations = activations * transformed_strengths
        result = self.softmax(sharp_activations)
        assert list(result.size())[1] == self._num_heads
        return result


class DNCMemory(nn.Module):
    def __init__(self, word_size=32, memory_size=100, num_reads=1, write_heads=1, batch_size=32):
        super(DNCMemory, self).__init__()
        self.W = word_size
        self.N = memory_size
        self.mem_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = write_heads
        self.batch_size = batch_size
        self.step = 0
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

        self._read_content_wght = WeightFn(word_size=word_size, num_heads=num_reads)
        self._write_content_wght = WeightFn(word_size=word_size, num_heads=write_heads)

        self._linkage = Linkage(memory_size=memory_size, num_writes=write_heads)
        self._usage = Usage(memory_size)

    def allocation(self, usage):
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

        sorted_nonusage, indices = nonusage.sort(-1, descending=True)
        _, perms = indices.sort(-1)

        sorted_usage = 1 - sorted_nonusage
        ones_ = _variable(torch.ones(usage.size(0), 1))
        x_base = torch.cat([ones_, sorted_usage], -1)

        prod_sorted_usage = x_base.cumprod(-1)
        sorted_allocation = sorted_nonusage * prod_sorted_usage[:, :-1]
        indexed = sorted_allocation.gather(-1, perms)

        return indexed


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
            allocation_weights.append(self.allocation(usage))
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
            usage: Current memory usage, which is a tensor of shape `[batch_size, memory_size]`,
            used for allocation-based addressing.

            Returns:
            tensor of shape `[batch_size, num_writes, memory_size]` indicating where
                to write to (if anywhere) for each write head.
            """

        alloc_gate, write_str, write_key, write_gate = inputs
        # c_t^{w, i} - The content-based weights for each write head.
        write_contnt_wghts = self._write_content_wght([memory, write_key, write_str])

        # a_t^i - The allocation weights for each write head.
        write_alloc_wghts = self.write_allocation_weights(usage, write_gate)

        assert list(memory.size()) == [self.batch_size, self.mem_size, self.word_size]
        assert list(write_contnt_wghts.size()) == [self.batch_size, self.num_writes, self.mem_size]
        assert list(write_alloc_wghts.size()) == [self.batch_size, self.num_writes, self.mem_size]

        # Expands gates over , [self.num_writes, self.word_size]memory locations.
        alloc_gate = alloc_gate.unsqueeze(-1)
        write_gate = write_gate.unsqueeze(-1)

        # w_t^{w, i} - The write weightings for each write head.
        result = write_gate * (alloc_gate * write_alloc_wghts + (1 - alloc_gate) * write_contnt_wghts)

        assert list(result.size()) == [self.batch_size, self.num_writes, self.mem_size]
        return result

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
        result = expanded_read_weights.bmm(link)
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

            prev_read_weights: A tensor `[batch_size, num_reads, memory_size]`
                containing the previous read locations.

            link: A tensor `[batch_size, num_writes, memory_size, memory_size]`
                containing the temporal write transition graphs.

            Returns:
            A tensor of shape `[batch_size, num_reads, memory_size]` containing the
            read weights for each read head.
            """
        read_keys, read_str, read_modes = inputs
 
        assert list(memory.size()) == [self.batch_size, self.mem_size, self.word_size]
        assert list(prev_read_weights.size()) == [self.batch_size, self.num_reads, self.mem_size]
        assert list(link.size()) == [self.batch_size, self.num_writes, self.mem_size, self.mem_size]

        # c_t^{r, i} - The content weightings for each read head.
        content_weights = self._read_content_wght([memory, read_keys, read_str])
        assert list(content_weights.size()) == [self.batch_size, self.num_reads, self.mem_size]

        forward_weights = prev_read_weights.unsqueeze(1).matmul(link)
        backwrd_weights = prev_read_weights.unsqueeze(1).matmul(link.transpose(2, 3))

        forward_weights = forward_weights.permute(0, 2, 1, 3)
        backwrd_weights = backwrd_weights.permute(0, 2, 1, 3)

        backward_mode = read_modes[:, :, 0]
        forward_mode = read_modes[:, :, 1]
        content_mode = read_modes[:, :, 2]

        content_str = content_mode.unsqueeze(2) * content_weights
        forward_str = (forward_mode.unsqueeze(2).unsqueeze(2) * forward_weights).sum(2)
        backwrd_str = (backward_mode.unsqueeze(2).unsqueeze(2) * backwrd_weights).sum(2)
        res = content_str + forward_str + backwrd_str
        assert list(res.size()) == [self.batch_size, self.num_reads, self.mem_size]
        return res

    def _erase_and_write(self, memory, address, erase_vec, values):
        """Module to erase and write in the external memory.

            Erase operation:
                M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

            Add operation:
                M_t(i) = M_t'(i) + w_t(i) * a_t

            where e are the erase_vec, w the write weights and a the values.

            Args:
                memory: 3-D tensor of   `[batch_size, memory_size, word_size]`.
                address: 3-D tensor     `[batch_size, num_writes, memory_size]`.
                erase_vec: 3-D tensor   `[batch_size, num_writes, word_size]`.
                values: 3-D tensor      `[batch_size, num_writes, word_size]`.

            Returns:
                3-D tensor of shape `[batch_size, num_writes, word_size]`.
            """
        assert list(memory.size()) == [self.batch_size, self.mem_size, self.word_size]
        assert list(address.size()) == [self.batch_size, self.num_writes, self.mem_size]
        assert list(erase_vec.size()) == [self.batch_size, self.num_writes, self.word_size]
        assert list(values.size()) == [self.batch_size, self.num_writes, self.word_size]

        expand_address = address.unsqueeze(3)
        erase_vec = erase_vec.unsqueeze(2)

        weighted_resets = expand_address * erase_vec
        weighted_resets = 1 - weighted_resets
        reset_gate = weighted_resets.prod(1)

        memory = memory * reset_gate
        add_matrix = address.transpose(1, 2).matmul(values)
        return memory.add(add_matrix)

    def forward(self,
                inputs, memory, read_weights, write_weights, links, link_weights, usage):
        """ forward
            """
        self.step += 1
        #if self.step % sl.log_step == 0:
            #sl.log_interface(inputs, self.step)

        [read_keys, read_str, write_key, write_str,
         erase_vec, write_vec, free_gates, alloc_gate, write_gate, read_modes] = inputs

        assert list(memory.size()) == [self.batch_size, self.N, self.W]

        # forward pass
        # 1 Update usage using 'free_gate' and previous read & write weights.
        nusage = self._usage(write_weights, free_gates, read_weights, usage)

        # 2 Get Write weights
        write_inputs = [alloc_gate, write_str, write_key, write_gate]
        nwrite_weights = self._write_weights(write_inputs, memory, nusage)

        #sl.log_if('access.calcs.write_weights2', nwrite_weights)
        # 3 Write to memory.
        memory = self._erase_and_write(memory, nwrite_weights, erase_vec, write_vec)
        #sl.log_if('access.calc.memory_out', memory)

        # 4 update linkage
        nlinks, nlink_weights = self._linkage(nwrite_weights, links, link_weights)

        # 5 read from memory
        read_inputs = [read_keys, read_str, read_modes]
        nread_weights = self._read_weights(read_inputs, memory, read_weights, nlinks)

        # 6 read memory
        read_words = read_weights.matmul(memory)

        return read_words, memory, nread_weights, nwrite_weights, nlinks, nlink_weights, nusage


class InterFace(nn.Module):
    def __init__(self, word_size=32, intf_size=100, num_reads=1, write_heads=1):
        super(InterFace, self).__init__()
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = write_heads
        self.batch_softmax = BatchSoftmax()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def _chunk1(self, inputs, dim1, activation=None):
        output = inputs[:, :dim1]
        rest = inputs[:, dim1:]
        if activation is not None:
            output = activation(output)
        return output, rest

    def _chunk2(self, inputs, dim1, dim2, activation=None):
        num = dim2 * dim1
        output = inputs[:, 0:num]
        rest = inputs[:, num:]
        output = output.contiguous().view(-1, dim1, dim2)
        if activation is not None:
            output = activation(output)
        return output, rest

    def chunk(self, inputs, dim, activation=None):
        if len(dim) == 1:
            return self._chunk1(inputs, dim[0], activation)
        else:
            dim1, dim2 = dim
            return self._chunk2(inputs, dim1, dim2, activation)

    def forward(self, inputs):
        # sl.log_if("interface.0in", inputs)

        read_keys, rest = self.chunk(inputs, [self.num_reads, self.word_size])
        read_strn, rest = self.chunk(rest, [self.num_reads])

        write_key, rest = self.chunk(rest, [self.num_writes, self.word_size])
        write_str, rest = self.chunk(rest, [self.num_writes])

        # e_t^i - Amount to erase the memory by before writing, for each write head.
        erase_vec, rest = self.chunk(rest, [self.num_writes, self.word_size], self.sigmoid)
        # v_t^i - The vectors to write to memory, for each write head `i`.
        write_vec, rest = self.chunk(rest, [self.num_writes, self.word_size])
        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        free_gates, rest = self.chunk(rest, [self.num_reads], self.sigmoid)
        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        alloc_gate, rest = self.chunk(rest, [self.num_writes], self.sigmoid)
        # g_t^{w, i} - Overall gating of write amount for each write head.
        write_gate, rest = self.chunk(rest, [self.num_writes], self.sigmoid)

        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        read_modes, _ = self.chunk(rest, [self.num_reads, 3], self.batch_softmax)

        return [read_keys, read_strn, write_key, write_str,
                erase_vec, write_vec, free_gates, alloc_gate,
                write_gate, read_modes]


class InterFace_Stateful(nn.Module):
    def __init__(self, word_size=32, intf_size=100, num_reads=1, write_heads=1):
        super(InterFace_Stateful, self).__init__()
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = write_heads
        self.num_interface = intf_size
        #
        self.batch_softmax = BatchSoftmax()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        #
        self.read_key = nn.Linear(intf_size, num_reads * word_size)
        self.read_strn = nn.Linear(intf_size, num_reads)
        #
        self.write_key = nn.Linear(intf_size, write_heads * word_size)
        self.write_str = nn.Linear(intf_size, write_heads)
        #
        self.erase_vec = nn.Linear(intf_size, write_heads * word_size)
        self.write_vec = nn.Linear(intf_size, write_heads * word_size)
        #
        self.free_gates = nn.Linear(intf_size, num_reads)
        self.alloc_gate = nn.Linear(intf_size, write_heads)
        self.write_gate = nn.Linear(intf_size, write_heads)
        #
        self.read_modes = nn.Linear(intf_size, num_reads * 3)

    def forward(self, inputs):
        read_keys = self.read_key(inputs)
        read_strn = self.read_strn(inputs)

        write_key = self.write_key(inputs)
        write_str = self.write_str(inputs)

        # e_t^i - Amount to erase the memory by before writing, for each write head.
        erase_vec = self.sigmoid(self.erase_vec(inputs))
        # v_t^i - The vectors to write to memory, for each write head `i`.
        write_vec = self.write_vec(inputs)

        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        free_gates = self.sigmoid(self.free_gates(inputs))
        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        alloc_gate = self.sigmoid(self.alloc_gate(inputs))
        # g_t^{w, i} - Overall gating of write amount for each write head.
        write_gate = self.sigmoid(self.write_gate(inputs))

        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        read_modes = self.batch_softmax(self.read_modes(inputs))

        print(read_modes.size())
        return [read_keys, read_strn, write_key, write_str,
                erase_vec, write_vec, free_gates, alloc_gate,
                write_gate, read_modes]


class DNC(nn.Module):
    def __init__(self,
                 batch_size=32,
                 memory_size=100,
                 word_len=6,
                 output_size=None,
                 num_layers=2,
                 hidden_size=4,
                 num_read_heads=1,
                 **kwdargs):
        super(DNC, self).__init__()
        self.mem_size = memory_size
        self.batch_size = batch_size
        self.word_len = word_len
        self.num_reads = num_read_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_writes = 1 # default from paper - not messin widdit
        self.interface_size = num_read_heads * word_len + 3 * word_len + 5 * num_read_heads + 3
        self.output_size = output_size if output_size is not None else word_len
        self._interface = InterFace(word_size=word_len,
                                    num_reads=num_read_heads,
                                    intf_size=self.interface_size + word_len,
                                    write_heads=1)
        self.controller = Controller(word_size=word_len,
                                     input_size=self.output_size + word_len * num_read_heads, # ok
                                     num_reads=num_read_heads,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     output_size=self.interface_size + word_len, # ???????
                                     batch_size=batch_size)
        self._memory = DNCMemory(word_size=word_len,
                                 memory_size=memory_size,
                                 batch_size=batch_size,
                                 num_reads=num_read_heads)
        self.nn_output = nn.Linear(self.interface_size + word_len + num_read_heads * word_len, self.output_size)


    def init_state(self, grad=True):
        """ prev_state: A `DNCState` tuple containing the fields
            `access_output` `[batch_size, num_reads, word_size]` containing read words.
            `access_state` is a tuple of the access module's state
            `controller_state` is a tuple of controller module's state
            """
        return [_variable(torch.zeros(self.batch_size, self.num_reads, self.word_len), requires_grad=grad),
                _variable(torch.zeros(self.batch_size, self.mem_size, self.word_len), requires_grad=grad), #memory
                _variable(torch.zeros(self.batch_size, self.num_reads, self.mem_size), requires_grad=grad), #read_weights
                _variable(torch.zeros(self.batch_size, self.num_writes, self.mem_size), requires_grad=grad), #write weights
                _variable(torch.zeros(self.batch_size, self.num_writes, self.mem_size, self.mem_size), requires_grad=grad),  #linkage
                _variable(torch.zeros(self.batch_size, self.num_writes, self.mem_size), requires_grad=grad), #linkage weight
                _variable(torch.zeros(self.batch_size, self.mem_size), requires_grad=grad)]

    def init_rnn(self, grad=True):
        return [_variable(torch.rand(self.num_layers, self.batch_size, self.hidden_size).float()),
                _variable(torch.rand(self.num_layers, self.batch_size, self.hidden_size).float())]

    @property
    def memory(self):
        return self._memory



    def forward(self, inputs, prev_controller_state, previous_state):
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
        o, m, rw, ww, l, lw, u = previous_state
        prev_access_output = o.view(-1, self.num_reads * self.word_len)

        control_input = torch.cat([inputs, prev_access_output], 1)
        control_output, controller_state = self.controller(control_input, prev_controller_state)
        interface_output = self._interface(control_output)

        access_output, m, rw, ww, l, lw, u = self._memory(interface_output, m, rw, ww, l, lw, u)
        access_output_v = access_output.view(-1, self.word_len * self.num_reads)

        output = torch.cat([control_output, access_output_v], 1)
        output_f = self.nn_output(output)

        return output_f, [access_output_v, m, rw, ww, l, lw, u], controller_state


def require_nonleaf_grad(v):
    def hook(g):
        v.grad_nonleaf = g
    v.register_hook(hook)


if __name__ == "__main__":
    print("x")
