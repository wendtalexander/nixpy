import time
from IPython import embed
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, wait
import nixio
from nixio.exceptions import DuplicateName
import numpy as np


def main(assigned_chunks):
    total_rows = 42_000_000
    num_cols = 32
    chunk_rows = 60_000

    np.random.seed(42)
    with nixio.File("large_dataset.nix", mpi=True) as file:
        try:
            block = file.create_block("data", "mult")
        except DuplicateName:
            block = file.blocks[0]

        try:
            dset = block.create_data_array(
                "data_array",
                "int16",
                dtype=nixio.DataType.Int16,
                shape=(total_rows, num_cols),
            )
        except DuplicateName:
            dset = block.data_arrays["data_array"]

        for chunk_idx in assigned_chunks:
            start = chunk_idx * chunk_rows
            end = min(start + chunk_rows, total_rows)
            dset[start:end, :] = np.random.randn(end - start, 32)


if __name__ == "__main__":
    start = time.time()
    total_rows = 42_000_000
    num_cols = 32
    workers = 8
    chunk_rows = 60_000

    # Calculate the total number of chunks
    num_chunks = total_rows // chunk_rows
    if total_rows % chunk_rows != 0:
        num_chunks += 1

    # Assign chunks to processes based on the desired pattern
    # For example, every 2nd chunk to process 2, every 3rd chunk to process 3, etc.
    chunk_assignments = {}
    for chunk_idx in range(num_chunks):
        # Determine which process should handle this chunk
        # For example, assign chunk 0 to process 0, chunk 1 to process 1, chunk 2 to process 2, etc.
        # Use modulo to cycle through processes

        assigned_process = chunk_idx % workers
        if assigned_process not in chunk_assignments:
            chunk_assignments[assigned_process] = []
        chunk_assignments[assigned_process].append(chunk_idx)
    with MPIPoolExecutor(max_workers=workers) as executor:
        P = executor.num_workers
        res = [executor.submit(main, chunk_assignments[i]) for i in range(P)]
        wait(res)
        end = time.time()
        duration = end - start
        print(f"process finished in {duration} s")

    embed()
    exit()
