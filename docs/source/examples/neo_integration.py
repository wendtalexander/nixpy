#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from IPython import embed
from neo.io import NixIO

import nixio

file_name = "neo_integration.nix"

# create a new file overwriting any existing content
nixfile = nixio.File.open(file_name, nixio.FileMode.Overwrite)
# create a new block
block = nixfile.create_block("ephys-recording", "neo.segment")
# we have to create a group with type neo.segmentb
recording_group = block.create_group("recording.data", "neo.segment")
section = nixfile.create_section("recording.metadata", "neo.segment")
channel_metadata = section.create_section("Channel.metadata", "neo.analogsingal")
channel_metadata.create_property("nix_name", "recording")
channel_metadata.create_property("t_start", 0.0)

# here you can append your data arrays to your group

for ch in range(16):
    data_array = block.create_data_array(
        f"channel {ch}",
        "neo.analogsignal",
        data=np.random.randint(-10, 10, size=1_000),
        unit="uV",
        label="Voltage",
    )
    data_array.append_sampled_dimension(1 / 30_000, label="time", unit="s", offset=0)
    data_array.metadata = channel_metadata
    recording_group.data_arrays.append(data_array)
d = NixIO(file_name).read()
