# -*- coding: utf-8 -*-
# Copyright © 2014, German Neuroinformatics Node (G-Node)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted under the terms of the BSD License. See
# LICENSE file in the root of the Project.
import gc
import os
import shutil
import time
import unittest

import h5py
import numpy as np
import pytest

MPI = pytest.importorskip("mpi4py.MPI")

import nixio as nix
import nixio.file as filepy
from nixio.exceptions import DuplicateName, InvalidFile


@pytest.fixture
def nix_file_factory(mpi_tmp_path, request):
    def _factory(comm, info, mode):
        nix_file_path = mpi_tmp_path / "testfile.nix"
        return nix.File(nix_file_path, mode, mpi=True, mpi_comm=comm, mpi_info=info)

    def cleanup():
        for file in mpi_tmp_path.iterdir():
            try:
                file.unlink()
            except Exception:
                pass

    request.addfinalizer(cleanup)

    return _factory


@pytest.mark.mpi
@pytest.mark.parametrize(
    "comm,info",
    [
        (MPI.COMM_WORLD, MPI.Info.Create()),
        # Add more (comm, info) pairs for this test
    ],
)
def test_file_format(nix_file_factory, comm, info):
    with nix_file_factory(comm, info, mode=nix.FileMode.Overwrite) as f:
        assert f.format == "nix"


@pytest.mark.mpi
@pytest.mark.parametrize(
    "comm,info",
    [
        (MPI.COMM_WORLD, MPI.Info.Create()),
        (MPI.COMM_WORLD.Split(color=0, key=0), MPI.Info.Create()),
        # You can set custom info hints as well:
        # (MPI.COMM_WORLD, custom_info),
    ],
)
def test_file_format_different_comm(nix_file_factory, comm, info):
    # Optionally set info hints
    info.Set("cb_buffer_size", "1048576")
    with nix_file_factory(comm, info, mode=nix.FileMode.Overwrite) as f:
        assert f.format == "nix"


@pytest.mark.mpi
def test_two_blocks_per_rank(nix_file_factory):
    comm = MPI.COMM_WORLD
    info = MPI.Info.Create()
    rank = comm.Get_rank()
    size = comm.Get_size()

    with nix_file_factory(comm, info, mode=nix.FileMode.Overwrite) as f:
        # Create two blocks for this rank
        for r in range(size):
            f.create_block(f"block_{r}_1", "test")
            f.create_block(f"block_{r}_2", "test")
        comm.Barrier()

        # Optionally, check that the blocks exist
        if rank == 0:
            blocks = f.blocks
            for r in range(size):
                assert blocks[f"block_{r}_1"]
                assert blocks[f"block_{r}_2"]

            assert len(blocks) == 2 * size


@pytest.mark.mpi
def test_dataarray_parallel_write(nix_file_factory):
    comm = MPI.COMM_WORLD
    info = MPI.Info.Create()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create the file and DataArray (all ranks participate)
    with nix_file_factory(comm, info, mode=nix.FileMode.Overwrite) as f:
        # All ranks must call create_block and create_data_array with the same arguments!
        block = f.create_block("block", "test")
        data_shape = (size, 5)
        data = np.zeros(data_shape, dtype=np.float64)
        da = block.create_data_array(
            "parallel_data", "test", shape=data_shape, dtype=np.float64
        )
        my_data = np.full((5,), fill_value=rank, dtype=np.float64)
        da[rank, :] = my_data

        comm.Barrier()

        # Optionally, check the data (from rank 0)
        if rank == 0:
            arr = da[:]
            for r in range(size):
                assert np.all(arr[r, :] == r)


def set_header(h5root, fformat=None, version=None, fileid=None):
    if fformat is None:
        fformat = filepy.FILE_FORMAT
    if version is None:
        version = filepy.HDF_FF_VERSION
    if fileid is None:
        fileid = nix.util.create_id()
    h5root.attrs["format"] = fformat
    h5root.attrs["version"] = version
    h5root.attrs["id"] = fileid
    h5root.attrs["created_at"] = 0
    h5root.attrs["updated_at"] = 0
    if "data" not in h5root:
        h5root.create_group("data")
        h5root.create_group("metadata")


@pytest.mark.mpi
def test_read_write(mpi_tmp_path):
    comm = MPI.COMM_WORLD
    info = MPI.Info.Create()
    rank = comm.Get_rank()
    size = comm.Get_size()

    testfilename = mpi_tmp_path / "vertest.nix"
    # Create the file and set the header
    with h5py.File(
        testfilename, mode="w", driver="mpio", comm=comm, info=info
    ) as h5file:
        h5root = h5file["/"]
        set_header(h5root)
    with nix.File.open(
        testfilename, nix.FileMode.ReadWrite, mpi=True, mpi_comm=comm, mpi_info=info
    ) as nix_file:
        assert nix_file._h5file.attrs["format"] == filepy.FILE_FORMAT
        assert tuple(nix_file._h5file.attrs["version"]) == filepy.HDF_FF_VERSION
        assert "id" in nix_file._h5file.attrs
        assert "data" in nix_file._h5file
        assert "metadata" in nix_file._h5file


# import shutil
# import tempfile
# import unittest
# from pathlib import Path
#
# from mpi4py import MPI
#
# import nixio as nix  # or import nix if that's the correct import

# class TestNixFileFormat(unittest.TestCase):
#     def setUp(self):
#         # Create a temporary directory for each test
#         self.tmp_dir = tempfile.mkdtemp()
#         self.mpi_tmp_path = Path(self.tmp_dir)
#
#     def tearDown(self):
#         # Cleanup: remove all files in the temp directory
#         shutil.rmtree(self.tmp_dir)
#
#     def nix_file_factory(self, comm, info, mode):
#         nix_file_path = self.mpi_tmp_path / "testfile.nix"
#         return nix.File(nix_file_path, mode, mpi=True, mpi_comm=comm, mpi_info=info)
#
#     def test_file_format(self):
#         comm_info_pairs = [
#             (MPI.COMM_WORLD, MPI.Info.Create()),
#             # Add more (comm, info) pairs as needed
#         ]
#         for comm, info in comm_info_pairs:
#             with self.subTest(comm=comm, info=info):
#                 with self.nix_file_factory(
#                     comm, info, mode=nix.FileMode.Overwrite
#                 ) as f:
#                     self.assertEqual(f.format, "nix")

#
# class TestDataArrayParallelWrite(unittest.TestCase):
#     def setUp(self):
#         self.comm = MPI.COMM_WORLD
#         self.rank = self.comm.Get_rank()
#         self.size = self.comm.Get_size()
#         # Only rank 0 sets the file path
#         if self.rank == 0:
#             self.file_path = "/tmp/test_parallel_dataarray.nix"
#         else:
#             self.file_path = None
#         # Broadcast the file path to all ranks
#         self.file_path = self.comm.bcast(self.file_path, root=0)
#
#     def tearDown(self):
#         self.comm.Barrier()  # Ensure all ranks are done before cleanup
#         if self.rank == 0 and os.path.exists(self.file_path):
#             os.remove(self.file_path)
#
#     def test_dataarray_parallel_write(self):
#         info = MPI.Info.Create()
#         # All ranks participate in file creation and writing
#         with nix.File(
#             self.file_path,
#             nix.FileMode.Overwrite,
#             mpi=True,
#             mpi_comm=self.comm,
#             mpi_info=info,
#         ) as f:
#             block = f.create_block("block", "test")
#             data_shape = (self.size, 5)
#             data = np.zeros(data_shape, dtype=np.float64)
#             da = block.create_data_array(
#                 "parallel_data", "test", shape=data_shape, dtype=np.float64
#             )
#             my_data = np.full((5,), fill_value=self.rank, dtype=np.float64)
#             da[self.rank, :] = my_data
#
#             self.comm.Barrier()
#
#             # Only rank 0 checks the data
#             if self.rank == 0:
#                 arr = da[:]
#                 for r in range(self.size):
#                     self.assertTrue(np.all(arr[r, :] == r))
#
#
# #
# #
# if __name__ == "__main__":
#     unittest.main()
