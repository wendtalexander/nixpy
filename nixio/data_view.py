# -*- coding: utf-8 -*-
# Copyright © 2014, German Neuroinformatics Node (G-Node)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted under the terms of the BSD License. See
# LICENSE file in the root of the Project.
from numbers import Integral
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from .data_set import DataSet
from .exceptions import OutOfBounds, IncompatibleDimensions


class DataView(DataSet):

    def __init__(self, da, slices):
        if len(slices) != len(da.shape):
            # This is always checked by the calling function, but we repeat
            # the check here for future bug catching
            raise IncompatibleDimensions(
                "Number of dimensions for DataView does not match underlying "
                "data object: {} != {}".format(len(slices), len(da.shape)),
                "DataView"
            )

        if any(s.stop > e for s, e in zip(slices, da.data_extent)):
            raise OutOfBounds(
                "Trying to create DataView which is out of bounds of the "
                "underlying DataArray"
            )

        # Simplify all slices
        slices = tuple(slice(*sl.indices(dimlen))
                       for sl, dimlen in zip(slices, da.shape))

        self.array = da
        self._h5group = self.array._h5group
        self._slices = slices

    @property
    def data_extent(self):
        return tuple(s.stop - s.start for s in self._slices)

    @data_extent.setter
    def data_extent(self, v):
        raise AttributeError("can't set attribute")

    @property
    def data_type(self):
        return self.array.data_type

    def _write_data(self, data, sl=None):
        tsl = self._slices
        if sl:
            tsl = self._transform_coordinates(sl)
        super(DataView, self)._write_data(data, tsl)

    def _read_data(self, sl=None):
        tsl = self._slices
        if sl is not None:
            tsl = self._transform_coordinates(sl)
        return self.array._read_data(tsl)

    def _transform_coordinates(self, user_slices):
        """
        Takes a series (tuple) of slices or indices passed to the DataView and
        transforms them to the equivalent slices or indices for the underlying
        DataArray. Bounds checking is performed on the results to make sure it
        is not outside the DataView's range.

        Note: HDF5 hyperslabs don't support negative steps, so we catch it
        here to throw an error from NIX instead to shorten the stack trace (we
        use the same message).
        """
        oob = OutOfBounds("Trying to access data outside of range of DataView")

        def transform_slice(uslice, dvslice):
            """
            Single dimension transform function for slices.

            uslice: User provided slice for dimension
            dvslice: DataView slice for dimension
            """
            # Simplify uslice; DataView step is always 1
            dimlen = dvslice.stop - dvslice.start
            ustart, ustop, ustep = uslice.indices(dimlen)
            if ustop < 0:  # special case for None stop
                ustop = dimlen + ustop
            tslice = slice(dvslice.start+ustart, dvslice.start+ustop, ustep)
            if tslice.stop > dvslice.stop:
                raise oob

            if tslice.step < 0:
                raise ValueError("Step must be >= 1")

            return tslice

        dvslices = self._slices
        user_slices = self._expand_user_slices(user_slices)
        tslices = list()
        for uslice, dvslice in zip(user_slices, dvslices):
            if isinstance(uslice, Integral):
                if uslice < 0:
                    tslice = dvslice.stop + uslice
                else:
                    tslice = uslice + dvslice.start
                if tslice < dvslice.start or tslice >= dvslice.stop:
                    raise oob
            elif isinstance(uslice, slice):
                tslice = transform_slice(uslice, dvslice)
                if tslice.start < dvslice.start:
                    raise oob
                if tslice.stop > dvslice.stop:
                    raise oob
            else:
                raise TypeError("Data indices must be integers or slices, "
                                "not {}".format(type(uslice)))
            tslices.append(tslice)

        return tuple(tslices)

    def _expand_user_slices(self, user_slices):
        """
        Given the user-supplied slices or indices, expands Ellipses if
        necessary and returns the same objects in a tuple padded with
        slice(None) to match the dimensionality of the DataView.
        """
        if not isinstance(user_slices, Iterable):
            user_slices = (user_slices,)

        if user_slices.count(Ellipsis) > 1:
            raise IndexError(
                "an index can only have a single ellipsis ('...')"
            )
        elif user_slices.count(Ellipsis) == 1:
            # expand slices at Ellipsis index
            expidx = user_slices.index(Ellipsis)
            npad = len(self.data_extent) - len(user_slices) + 1
            padding = (slice(None),) * npad
            return user_slices[:expidx] + padding + user_slices[expidx+1:]

        # expand slices at the end
        npad = len(self.data_extent) - len(user_slices)
        padding = (slice(None),) * npad
        return user_slices + padding
