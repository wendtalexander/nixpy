from nixio.exceptions.exceptions import InvalidSlice
import os
import time
import unittest
from collections import OrderedDict
from IPython import embed
import numpy as np
import nixio as nix
from nixio.exceptions import DuplicateName, UnsupportedLinkType
from nixio.test.tmp import TempDir


class TestMultiTagSlice(unittest.TestCase):
    def setUp(self):
        interval = 1.0
        ticks = [1.2, 2.3, 3.4, 4.5, 6.7]
        unit = "ms"

        self.tmpdir = TempDir("mtagtest")
        self.testfilename = os.path.join(self.tmpdir.path, "mtagtest.nix")
        self.file = nix.File.open(self.testfilename, nix.FileMode.Overwrite)
        self.block = self.file.create_block("test block", "recordingsession")

        self.my_array = self.block.create_data_array(
            "my array", "test", nix.DataType.Int16, (0, 0)
        )
        self.my_tag = self.block.create_multi_tag("my tag", "tag", self.my_array)

        self.your_array = self.block.create_data_array(
            "your array", "test", nix.DataType.Int16, (0, 0)
        )
        self.your_tag = self.block.create_multi_tag("your tag", "tag", self.your_array)

        self.data_array = self.block.create_data_array(
            "featureTest", "test", nix.DataType.Double, (2, 10, 5)
        )

        data = np.zeros((2, 10, 5))
        value = 0.0
        for i in range(2):
            value = 0
            for j in range(10):
                for k in range(5):
                    value += 1
                    data[i, j, k] = value

        self.data_array[:, :, :] = data

        set_dim = self.data_array.append_set_dimension()
        set_dim.labels = ["label_a", "label_b"]
        sampled_dim = self.data_array.append_sampled_dimension(interval)
        sampled_dim.unit = unit
        range_dim = self.data_array.append_range_dimension(ticks)
        range_dim.unit = unit

        event_positions = np.zeros((2, 3))
        event_positions[0, 0] = 0.0
        event_positions[0, 1] = 3.0
        event_positions[0, 2] = 3.4

        event_positions[1, 0] = 0.0
        event_positions[1, 1] = 8.0
        event_positions[1, 2] = 2.3

        event_extents = np.zeros((2, 3))
        event_extents[0, 0] = 1.0
        event_extents[0, 1] = 6.0
        event_extents[0, 2] = 2.3

        event_extents[1, 0] = 1.0
        event_extents[1, 1] = 3.0
        event_extents[1, 2] = 2.0

        event_labels = ["event 1", "event 2"]
        dim_labels = ["dim 0", "dim 1", "dim 2"]

        self.event_array = self.block.create_data_array(
            "positions", "test", data=event_positions
        )

        self.extent_array = self.block.create_data_array(
            "extents", "test", data=event_extents
        )
        extent_set_dim = self.extent_array.append_set_dimension()
        extent_set_dim.labels = event_labels
        extent_set_dim = self.extent_array.append_set_dimension()
        extent_set_dim.labels = dim_labels

        self.feature_tag = self.block.create_multi_tag(
            "feature_tag", "events", self.event_array
        )
        self.feature_tag.extents = self.extent_array
        self.feature_tag.references.append(self.data_array)

        embed()
        exit()


if __name__ == "__main__":
    test = TestMultiTagSlice()
    test.setUp()
