from dataset_loader import load_CSVs, select_paths, load_content, load_pair, load_dataset
from keras.utils import Sequence
import multiprocessing
import numpy as np
import time
import os


class DataLoader(Sequence):
    """Use when batch_size is equal to 1!"""

    def __init__(self, type: str = None, batch_size: int = 1):
        if type is None:
            raise Exception("Type hasn't been specified.")

        self.content = load_content(select_paths(load_CSVs(), type), ";")
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.content)//self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'

        #start = time.perf_counter_ns()

        X_pairs, y_pairs = load_dataset([self.content[index]])

        #stop = time.perf_counter_ns()
        #print(f"{(stop - start)/1e6 = } ms")

        return [X_pairs[:, 0, :, :], X_pairs[:, 1, :, :]], y_pairs


class ParallelDataLoader(Sequence):
    """Use when batch_size is more than 1!"""

    def __init__(self, type: str = None, batch_size: int = 16):
        if type is None:
            raise Exception("Type hasn't been specified.")

        self.content = load_content(select_paths(load_CSVs(), type), ";")
        self.batch_size = batch_size

        max_threads = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(
            batch_size if batch_size < (max_threads - 1) else (max_threads - 2)
        )

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.content) // self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        batch = self.content[index * self.batch_size: (index + 1) * self.batch_size]

        #start = time.perf_counter_ns()

        result = self.pool.starmap(load_pair, batch, 1)

        X_pairs, y_pairs = [], []
        for pairs, match in result:
            X_pairs.append(pairs)
            y_pairs.append(match)

        X_pairs = np.array(X_pairs)
        y_pairs = np.array(y_pairs)

        #stop = time.perf_counter_ns()
        #print(f"{(stop - start)/1e6 = } ms")

        return [X_pairs[:, 0, :, :], X_pairs[:, 1, :, :]], y_pairs

    def __del__(self):
        self.pool.close()


if __name__ == "__main__":
    os.chdir("hand_writings/")
    loader = DataLoader("test")
    X_pairs, y_pairs = loader.__getitem__(0)

    loader = ParallelDataLoader("test")
    X_pairs, y_pairs = loader.__getitem__(0)
