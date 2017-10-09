from torch.utils.data import Dataset
import numpy as np


class RandomData(Dataset):   
    def __init__(self,
                 num_seq=10, 
                 seq_len=6, 
                 iters=1000, 
                 seq_width=4):
        self.seq_width = seq_width
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.iters = iters
        
    def __getitem__(self, index):
        con = np.random.randint(0, self.seq_width, size=self.seq_len)
        seq = np.zeros((self.seq_len, self.seq_width))
        seq[np.arange(self.seq_len), con] = 1
        end = np.asarray([[-1] * self.seq_width])
        zer = np.zeros((self.seq_len, self.seq_width))
        return seq, zer

    def __len__(self):
        return self.iters


