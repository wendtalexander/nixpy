# -*- coding: utf-8 -*-
# Copyright © 2016, German Neuroinformatics Node (G-Node)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted under the terms of the BSD License. See
# LICENSE file in the root of the Project.
from nixio.data_array import DataArray
from .data_set import DataSet
from .entity import Entity


class VirtualDataArray(DataArray):
    def __init__(self, nixfile, nixparent, h5group):
        super(VirtualDataArray, self).__init__(nixfile, nixparent, h5group)
        self._sources = None
        self._dimensions = None

    @classmethod
    def create_new(cls, nixfile, nixparent, h5parent, layout, name, type_):
        newentity = Entity.create_new(nixfile, nixparent, h5parent, name,
                                      type_)
        newentity = cls(nixfile, nixparent, newentity._h5group)
        newentity._h5group.group.create_virtual_dataset("data", layout)

        return newentity
