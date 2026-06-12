import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd


class CEPixelIndexBuilder:
    """
    Builds CE_pixel_idx:
        CE_pixel_idx[i] = list of raster indices belonging to CE i
    """

    def __init__(self, ce_shapefile, raster_path):
        self.ce_shapefile = ce_shapefile
        self.raster_path = raster_path

    def build(self):
        # Load CE polygons
        ce = gpd.read_file(self.ce_shapefile)

        # Load raster
        with rasterio.open(self.raster_path) as src:
            raster_shape = (src.height, src.width)
            transform = src.transform

        CE_pixel_idx = []

        # Loop over CE polygons
        for i, row in ce.iterrows():
            geom = row.geometry

            # Rasterize polygon
            mask = features.rasterize(
                [(geom, 1)],
                out_shape=raster_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

            # Extract pixel indices
            idx = np.where(mask == 1)
            flat_idx = np.ravel_multi_index(idx, raster_shape)

            CE_pixel_idx.append(flat_idx)

        print("✓ CE_pixel_idx built")
        return CE_pixel_idx
