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
import time
import shutil
import h5py
import unittest
import numpy as np
from IPython import embed

import mpi4py
mpi_installed = True
from mpi4py import MPI

import nixio as nix
import nixio.file as filepy
from nixio.exceptions import InvalidFile, DuplicateName


MPI_ENABLED = getattr(h5py.get_config(), 'mpi', False)

from tempfile import mkdtemp


class TempDir:

    def __init__(self, prefix=None):
        self.path = mkdtemp(prefix=prefix)

    def cleanup(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def __del__(self):
        self.cleanup()

class TestFileVer(unittest.TestCase):

    backend = "h5py"
    filever = filepy.HDF_FF_VERSION
    fformat = filepy.FILE_FORMAT
    mpi=True
    comm=MPI.COMM_WORLD

    def try_open(self, mode):
        with nix.File(self.testfilename, mode, mpi=self.mpi) as file:
            pass

    def set_header(self, fformat=None, version=None, fileid=None):
        h5file = h5py.File(self.testfilename, mode="w", driver="mpio", comm=self.comm)
        h5root = h5file["/"]

        if fformat is None:
            fformat = self.fformat
        if version is None:
            version = self.filever
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
        h5file.close()

    def setUp(self):
        self.tmpdir = TempDir("vertest")
        self.testfilename = os.path.join(self.tmpdir.path, "vertest.nix")

    def tearDown(self):
        self.tmpdir.cleanup()

    # def test_read_write(self):
    #     self.set_header()
    #     self.try_open(nix.FileMode.ReadWrite)

    def test_read_only(self):
        ver_x, ver_y, ver_z = self.filever
        roversion = (ver_x, ver_y, ver_z+2)
        self.set_header(version=roversion)
        self.try_open(nix.FileMode.ReadOnly)
        with self.assertRaises(RuntimeError):
            self.try_open(nix.FileMode.ReadWrite)
    #
    # def test_no_open(self):
    #     ver_x, ver_y, ver_z = self.filever
    #     noversion = (ver_x, ver_y+3, ver_z+2)
    #     self.set_header(version=noversion)
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadWrite)
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadOnly)
    #     noversion = (ver_x, ver_y+1, ver_z)
    #     self.set_header(version=noversion)
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadWrite)
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadOnly)
    #     noversion = (ver_x+1, ver_y, ver_z)
    #     self.set_header(version=noversion)
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadWrite)
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadOnly)
    #
    # def test_bad_tuple(self):
    #     self.set_header(version=(-1, -1, -1))
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadOnly)
    #     self.set_header(version=(1, 2))
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadOnly)
    #
    # def test_bad_format(self):
    #     self.set_header(fformat="NOT_A_NIX_FILE")
    #     with self.assertRaises(InvalidFile):
    #         self.try_open(nix.FileMode.ReadOnly)
    #
    # def test_bad_id(self):
    #     self.set_header(fileid="")
    #     with self.assertRaises(RuntimeError):
    #         self.try_open(nix.FileMode.ReadOnly)
    #
    #     # empty file ID OK for versions older than 1.2.0
    #     self.set_header(version=(1, 1, 1), fileid="")
    #     self.try_open(nix.FileMode.ReadOnly)
    #
    #     self.set_header(version=(1, 1, 0), fileid="")
    #     self.try_open(nix.FileMode.ReadOnly)
    #
    #     self.set_header(version=(1, 0, 0), fileid="")
    #     self.try_open(nix.FileMode.ReadOnly)

if __name__ == "__main__":
    ts = TestFileVer()
    embed()
    exit()
