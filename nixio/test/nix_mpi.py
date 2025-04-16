import time
from mpi4py import MPI
import nixio
from nixio.exceptions import DuplicateName
from IPython import embed
import numpy as np


def main(dset, assigned_chunks, rank):
    for chunk_idx in assigned_chunks:
        start = chunk_idx * chunk_rows
        end = min(start + chunk_rows, total_rows)
        dset[start:end, :] = np.random.randn(end - start, 32)
    return dset


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    total_rows = 42_000_000
    num_cols = 32

    with nixio.File("large_dataset.nix", mpi=True) as file:
        start = time.time()
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

        end = time.time()
        duration = end - start
        print(f"Creation of dset took {duration} s")

        total_rows = dset.shape[0]
        num_cols = dset.shape[1]
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

            assigned_process = chunk_idx % size
            if assigned_process not in chunk_assignments:
                chunk_assignments[assigned_process] = []
            chunk_assignments[assigned_process].append(chunk_idx)

        assigned_chunks = chunk_assignments.get(rank, [])

        start = time.time()
        dset = main(dset, assigned_chunks, rank)
        end = time.time()
        duration = end - start
        print(f"process {rank} finished in {duration} s")
        # file.flush()
        comm.Barrier()
    #     # MPI.Finalize()
    # file.close()
