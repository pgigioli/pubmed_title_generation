import numpy as np

class Data_Batcher:
    """ This class stores a dataset and provides randomized batch production of data
    
    Attributes: 
        data (list)      : list of dataset samples
        batch_size (int) : number of samples to produce in a batch
        data_iterator    : iterates through the data in batches
            
    """  
    def __init__(self, data, batch_size):
        """ Loads dataset and creates a data iterator """
        self.data = data
        self.batch_size = batch_size
        self.epoch = 0
        self.data_iterator = self.make_random_iter()
        
    def next_batch(self):
        """ Produces a batch of samples with size of batch_size 
        
        Returns:
            list : list of data samples pulled from original dataset
            list : list of data indices that maps data samples to original dataset
            
        """
        try:
            idxs = next(self.data_iterator)
        except StopIteration:
            self.epoch += 1
            self.data_iterator = self.make_random_iter()
            idxs = next(self.data_iterator)
            
        batch = [self.data[i] for i in idxs]
        batch_idxs = [idx for idx in idxs]
        return batch, self.epoch

    def make_random_iter(self):
        """ Creates an iterator that randomly selects batches of data 
        
        Returns:
            iter : an iterator that generates random batches of data
        
        """
        if self.batch_size >= len(self.data):
            splits = np.array([len(self.data)])
        else:
            splits = np.arange(self.batch_size, len(self.data), self.batch_size)
        it = np.split(np.random.permutation(range(len(self.data))), splits)
        return iter(it)