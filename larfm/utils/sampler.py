
from torch.utils.data import Sampler, BatchSampler

class SkipBatchSampler(BatchSampler):
    """
    A BatchSampler that skips the first N batches of another BatchSampler.
    Useful for resuming training from a specific iteration without loading all previous data.
    """
    def __init__(self, batch_sampler, skip_batches=0):
        self.batch_sampler = batch_sampler
        self.skip_batches = skip_batches

    def __iter__(self):
        for i, batch in enumerate(self.batch_sampler):
            if i >= self.skip_batches:
                yield batch

    def __len__(self):
        return len(self.batch_sampler) - self.skip_batches

