import math
import warnings
from abc import ABC, abstractmethod

import geopandas as gpd
import h3.api.numpy_int as h3
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import cascaded_union

from skmob.utils import constants, utils


class TessellationTilers:
    def __init__(self):
        self._tilers = {}

    def register_tiler(self, key, tiler):
        self._tilers[key] = tiler

    def create(self, key, **kwargs):
        tiler = self._tilers.get(key)

        if not tiler:
            raise ValueError(key)

        return tiler(**kwargs)

    def get(self, service_id, **kwargs):
        return self.create(service_id, **kwargs)


tiler = TessellationTilers()


class TessellationTiler(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def _build(self, **kwargs):
        pass


class VoronoiTessellationTiler(TessellationTiler):
    def __init__(self):

        super().__init__()
        self._instance = None

    def __call__(self, points, crs=constants.DEFAULT_CRS):

        if not self._instance:

            if isinstance(points, gpd.GeoDataFrame):

                if not all(isinstance(x, Point) for x in points.geometry):
                    raise ValueError("Not valid points object. Accepted type is GeoDataFrame.")

        return self._build(points, crs)

    def _build(self, points, crs=constants.DEFAULT_CRS):

        gdf = gpd.GeoDataFrame(points.copy(), crs=crs)
        gdf.loc[:, constants.TILE_ID] = list(np.arange(0, len(gdf)))

        # Convert TILE_ID to have str type
        gdf[constants.TILE_ID] = gdf[constants.TILE_ID].astype("str")

        return gdf[[constants.TILE_ID, "geometry"]]


# Register the builder
tiler.register_tiler("voronoi", VoronoiTessellationTiler())