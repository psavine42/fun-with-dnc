import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

eps = 10e-6

def interface_part(num_reads, W):
                    #read_keys
    partition = [num_reads* W, num_reads, W, 1, W, W, num_reads, 1, 1, num_reads * 3]
    ds = []
    cntr = 0
    for idx in partition:
        tn = [cntr, cntr + idx]
        ds.append(tn)
        cntr += idx
    return ds

def oneplus(x):
    return 1 + torch.log(1 + e ** x)



def show(tensors, m=""):
    print("--")
    if type(tensors) == torch.Tensor:
        print(m, tensors.size())
    else:
        print(m)
        [print(t.size()) for t in tensors]


def erase_and_write(memory, address, erase_vec, values):
    """Module to erase and write in the external memory.

        Erase operation:
            M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

        Add operation:
            M_t(i) = M_t'(i) + w_t(i) * a_t

        where e are the erase_vec, w the write weights and a the values.

        Args:
            memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
            address: 3-D tensor `[batch_size, num_writes, memory_size]`.
            erase_vec: 3-D tensor `[batch_size, num_writes, word_size]`.
            values: 3-D tensor `[batch_size, num_writes, word_size]`.

        Returns:
            3-D tensor of shape `[batch_size, num_writes, word_size]`.
        """
    
    show([memory, address, erase_vec, values])
    #expand_address = address #.unsqueeze(3)
    print("memory    ", memory.size())
    print("address   ", address.size())
    print("erase_vec ", erase_vec.size())
    print("values    ", values.size())
    

    #reset_weights = reset_weights.unsqueeze(2)

    
    weighted_resets =  address * erase_vec
    weighted_resets = 1 - weighted_resets
    reset_gate = weighted_resets.prod(1)
    memory *= reset_gate

    add_matrix = address.bmm(values)
    memory += add_matrix
    return memory

"""
def allocation_weighting(usage_vec, batch_size, num_words ):
    #sorted usage - the usage vector sorted ascndingly
    #the original indices of the sorted usage vector
    sorted_usage_vec, free_list = torch.top_k(-1 * usage_vec)
    sorted_usage_vec *= -1
    cumprod = torch.bmm(sorted_usage_vec, 1)
    unorder = (1-sorted_usage_vec) * cumprod

    alloc_weights = tf.zeros([batch_size, num_words])
    I = tf.constant(np.identity(num_words, dtype=np.float32))
        
    #for each usage vec
    for pos, idx in enumerate(tf.unstack(free_list[0])):
            #flatten
        m = tf.squeeze(tf.slice(I, [idx, 0], [1, -1]))
        #add to weight matrix
        alloc_weights += m*unorder[0, pos]
        #the allocation weighting for each row in memory
    return tf.reshape(alloc_weights, [num_words, 1])





def content_lookup(self, memory, key, weights):
    norm_mem = memory.norm(p=2, dim=1)
    norm_key = key.norm(0)
    sim = norm_mem.dot(norm_key)
    #str is 1*1 or 1*R
    #returns similarity measure
    return F.softmax(sim * weights, 0)
"""