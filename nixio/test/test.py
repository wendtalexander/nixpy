import os
import unittest

import numpy as np
from IPython import embed

import nixio as nix
from nixio.test.tmp import TempDir


class TestVirtualDataArray(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TempDir("blocktest")
        self.testfilename = os.path.join(self.tmpdir.path, "blocktest.nix")
        self.file = nix.File.open(self.testfilename, nix.FileMode.Overwrite)

    def tearDown(self):
        self.file.close()
        self.tmpdir.cleanup()

    def create_virtual_layout(self):
        self.file.virtual_layout((10, 10), _dtype=nix.DataType.Int16)


if __name__ == "__main__":
    file = nix.File.open("test.nix", nix.FileMode.Overwrite)
    layout = file.create_virtual_layout((5, 5), nix.DataType.Int16)
    b = file.create_block("test", "test")
    das = b.create_data_array("test_arra", "test", data=np.ones(5))
    file.append_to_virtual_layout(layout, das, axis=0)
    file.append_to_virtual_layout(layout, das, axis=0)
    file.append_to_virtual_layout(layout, das, axis=0)
    file.append_to_virtual_layout(layout, das, axis=0)

    vsd = b.create_virtual_data_array(layout, "vsd", "virtual_type", label="Volt", unit="mV")

    embed()
    exit()
