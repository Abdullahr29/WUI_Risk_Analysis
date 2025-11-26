import pandas as pd
import geopandas as gpd
import numpy as np
import leafmap
import pickle
import tomllib
import json
import os
import rasterio
from shapely import wkt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterio.mask import mask
from rasterio.windows import Window
from shapely.affinity import affine_transform
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from affine import Affine  
from pathlib import Path

from Utils.io_utils import RemoteIO

def get_project_root(marker="Utils"):
    """
    Walk upward until we find the project root, identified by containing the 'Utils' folder.
    This works regardless of where code is executed from (script, notebook, HPC).
    """
    current = Path().resolve()

    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent

    raise RuntimeError("Could not find project root")

# Cache so we don't recompute
_PROJECT_ROOT = str(get_project_root())

def get_project_root_path() -> str:
    return _PROJECT_ROOT

def get_data_path() -> str:
    return _PROJECT_ROOT + "/Data/"

def get_temp_path() -> str:
    return _PROJECT_ROOT + "/Temporary_Files/"

def get_main_csv_path() -> str:
    return _PROJECT_ROOT + "/Data/xBD_WUI_Analysis.csv"

def get_config_path() -> str:
    return _PROJECT_ROOT + "/config.toml"


damage_colors = {
    "no-damage": "cyan",
    "minor-damage": "yellow",
    "major-damage": "orange",
    "destroyed": "red"
}

CLASS_MAP = {
    1: "Residential: Large buildings and homes",
    2: "Residential: Small outbuildings",
    3: "Residential: Informal settlement",
    4: "Residential: Cars",
    5: "Residential: Miscellaneous",
    6: "Vegetative: Trees",
    7: "Vegetative: Shrubs and bushes",
    8: "Vegetative: Grass/lawns/low fuel",
    9: "Vegetative: Dried shrubs",
    10: "Vegetative: Dried grass",
    11: "Non-combustible: Roads/pavements",
    12: "Non-combustible: Bare soil",
    13: "Non-combustible: Water body",
    14: "Non-combustible: Undeveloped/concrete/bare",
    15: "Non-combustible: Destroyed structure",
    16: "Non-combustible: Burnt vegetation",
    17: "Unclassified",
    18: "Shadow",
    19: "Cloud",
    20: "Smoke",
    21: "Multi-object",
}

class DataLoader:
    def __init__(self):

        with open(get_config_path(), "rb") as f:
            config = tomllib.load(f)
        
        self.data_config = config["datapaths"]

        on_remote = config["local_vars"]["on_remote"]

        self.on_remote = on_remote

        if self.on_remote:
            self.data_path = f"{self.data_config["remote_root"]}{self.data_config["data_root"]}"
        else:
            self.data_path = f"{self.data_config["home_root"]}{self.data_config["data_root"]}"

        self.xBD_data = pd.read_csv(get_main_csv_path())

        self.current_scene_id = None
        self.current_scene_info = None
        self.current_scene_image_path = None
        self.current_norm_image_path = None
        self.is_current_scene_pre = None

        self.remote_scene_paths = []
    
    def GetSceneFromID(self, scene_id, with_norm_image=True, pre_image=True):
        scene_info = self.xBD_data[self.xBD_data["scene_id"] == scene_id].iloc[0]

        return self._get_scene(scene_info, with_norm_image=with_norm_image, pre_image_path=pre_image)

    
    def GetSceneFromLoc(self, loc, with_norm_image=False, pre_image=True):
        scene_info = self.xBD_data.iloc[loc]
        
        return self._get_scene(scene_info, with_norm_image=with_norm_image, pre_image_path=pre_image)
    

    def InspectSceneTif(self, scene_id, pre_image=True):
        scene_info = self.xBD_data[self.xBD_data["scene_id"] == scene_id].iloc[0]
        if pre_image:
            scene_path = scene_info["pre_image_path"]
        else:
            scene_path = scene_info["post_image_path"]

        full_path = self._get_file_path(scene_path)
        inspect_tif(full_path)


    def PlotImageFromSceneID(self, scene_id=None, norm_image=True, pre_image=True, with_labels=True, pre_labels=False):
        if scene_id is None:
            if self.current_scene_id is None:
                raise ValueError("No scene is currently loaded. Please provide a scene_id.")
            scene_id = self.current_scene_id

        if scene_id != self.current_scene_id or self.current_scene_id is None or self.current_norm_image_path is None or self.is_current_scene_pre != pre_image:
            if norm_image:
                _, norm_image_path, full_path, scene_info, _ = self.GetSceneFromID(scene_id, with_norm_image=True, pre_image=pre_image)
                image_path = norm_image_path
            else:
                _, full_path, scene_info, _ = self.GetSceneFromID(scene_id, with_norm_image=False, pre_image=pre_image)
                image_path = full_path
        else:
            if norm_image:
                image_path = self.current_norm_image_path
            else:
                image_path = self.current_scene_image_path
            scene_info = self.current_scene_info
 
        with rasterio.open(image_path) as src:
            arr = src.read([1,2,3]) if src.count >= 3 else src.read()
            image = arr.transpose((1,2,0)) if arr.ndim==3 else arr

        if with_labels:
            if pre_labels:
                labels_path = scene_info["pre_label_path"]
            else:
                labels_path = scene_info["post_label_path"]
            full_labels_path = self._get_file_path(labels_path)

            with open(full_labels_path, 'r') as f:
                labels_dict = json.load(f)

            if pre_labels:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(image)  # your loaded and transposed image

                for feature in labels_dict['features']['xy']:
                    poly = wkt.loads(feature['wkt'])
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color='red', alpha=0.5, linewidth=0.5)
                
                plt.show()
                
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(image)

                for feature in labels_dict['features']['xy']:
                    damage_level = feature['properties'].get('subtype', 'no-damage')
                    poly = wkt.loads(feature['wkt'])
                    x, y = poly.exterior.xy
                    color = damage_colors.get(damage_level, 'gray')  # fallback if unknown
                    ax.plot(x, y, color=color, alpha=0.5, linewidth=0.5)

                legend_elements = [
                    Line2D([0], [0], color=color, lw=2, label=label.replace('-', ' ').capitalize())
                    for label, color in damage_colors.items()
                ]

                ax.legend(handles=legend_elements)
                plt.show()

            return image, labels_dict

        else:
            fig = plt.subplots(figsize=(10, 8))
            plt.imshow(image)
            plt.show()

            return image
        
        
    def PlotImageFromLoc(self, loc, norm_image=True, pre_image=True, with_labels=True, pre_labels=False):
        scene_info = self.xBD_data.iloc[loc]
        return self.PlotImageFromSceneID(scene_info["scene_id"], norm_image=norm_image, pre_image=pre_image, with_labels=with_labels, pre_labels=pre_labels)

    def PlotImagePolygons(self, idx=None, scene_id=None, pre_image=True):
        if scene_id is None:
            if self.current_scene_id is None:
                raise ValueError("No scene is currently loaded. Please provide a scene_id.")
            scene_id = self.current_scene_id

        if self.is_current_scene_pre != pre_image:
            self.GetSceneFromID(scene_id, with_norm_image=True, pre_image=pre_image)
        
        poly = show_in_pixel_space(self.current_norm_image_path, self.current_scene_polygons, idx=idx)

        return poly


    def ClearCurrentScene(self):
        if self.current_norm_image_path and os.path.exists(self.current_norm_image_path):
            os.remove(self.current_norm_image_path)
        
        if self.on_remote:
            for path in self.remote_scene_paths:
                if os.path.exists(path):
                    os.remove(path)
        
        self.remote_scene_paths = []
        self.current_scene_id = None
        self.current_scene_info = None
        self.current_scene_image_path = None
        self.current_norm_image_path = None
        self.is_current_scene_pre = None
    
    def UploadImagePolygons(self, polygon_path, is_xBD=True):
        source_path = self.data_config["xBD_path"] if is_xBD else self.data_config["maxar_path"]

        dest_path = f"{self.data_path}{source_path}{self.data_config["analysis_path"]}{polygon_path}"
        src_path = f"{get_temp_path()}{polygon_path}"

        if self.on_remote:
            RemoteIO.put_file(src_path, dest_path)
        else:
            os.replace(src_path, dest_path)

        os.remove(src_path)


    def _get_scene(self, scene_info, with_norm_image=False, pre_image_path=True):
        self.ClearCurrentScene()
        scene_polygon_gdf = None
        scene_analysis_dict = None
        norm_image_path = None
        
        self.current_scene_id = scene_info["scene_id"]
        self.current_scene_info = scene_info

        if pre_image_path:
            scene_path = scene_info["pre_image_path"]
        else:
            scene_path = scene_info["post_image_path"]
        full_path = self._get_file_path(scene_path)

        self.current_scene_image_path = full_path
        self.is_current_scene_pre = pre_image_path

        if scene_info["segmented"] == True:
            scene_polygon_path = scene_info["polygon_path"]

            with open(self._get_file_path(scene_polygon_path, is_image_or_label=False), 'rb') as f:
                scene_analysis_dict = pickle.load(f)

            scene_polygon_gdf = scene_analysis_dict.polygons.iloc[0]
            self.current_scene_polygons = scene_polygon_gdf

        if with_norm_image:
            norm_image_path = normalize_to_uint8_per_band(full_path, export=True, export_path=f"{get_temp_path()}{scene_info["scene_id"]}_norm_image.tif", return_path=True)
            self.current_norm_image_path = norm_image_path

        return scene_polygon_gdf, norm_image_path, full_path, scene_info, scene_analysis_dict
    
    def _get_file_path(self, scene_path, is_xBD=True, is_image_or_label=True):
        source_path = self.data_config["xBD_path"] if is_xBD else self.data_config["maxar_path"]
        type_path = self.data_config["images_path"] if is_image_or_label else self.data_config["analysis_path"]

        full_path = f"{self.data_path}{source_path}{type_path}{scene_path}"

        if self.on_remote:
            
            dest_path = f"{get_temp_path()}{os.path.basename(scene_path)}"

            if not os.path.exists(dest_path):
                dest_path = RemoteIO.get_file(full_path, dest_path)

            final_path = dest_path
            self.remote_scene_paths.append(dest_path)
        else:
            final_path = full_path

        return final_path

    
def inspect_tif(image_path):
    with rasterio.open(image_path) as src:
        print(f"\n📂 Inspecting: {image_path}")
        print("Shape (height, width):", (src.height, src.width))
        print("Number of bands:", src.count)
        res_x, res_y = src.res
        print("Pixel size (X):", res_x)
        print("Pixel size (Y):", res_y)
        print("CRS:", src.crs)
        print("Affine Transform:", src.transform)
        print("Band order and descriptions:")
        print()

        for i in range(1, src.count + 1):
            desc = src.descriptions[i-1] or f"Band {i}"
            dtype = src.dtypes[i-1]
            stats = src.read(i).min(), src.read(i).max()
            shape = src.read(i).shape
            print(f"  - Band {i}: {desc}, dtype: {dtype}, min-max: {stats}, shape: {shape}")

def normalize_to_uint8_per_band(image_path, export=False, export_path="norm_image.tif", return_path=False):
    """
    Normalizes a multi-band image to 0–255 per band and converts to uint8.
    Expects a path to a .tif file.
    If export=True, saves normalized image to a GeoTIFF at `export_path`.
    Returns: normalized image (H, W, C) as uint8
    """

    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])  # shape: (C, H, W)
        meta = src.meta.copy()

    # Transpose to (H, W, C)
    if image.shape[0] in [1, 2, 3, 4, 5]:  # (C, H, W) → (H, W, C)
        image = np.transpose(image, (1, 2, 0))

    norm_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[2]):
        band = image[:, :, i].astype(np.float32)
        band -= band.min()
        if band.max() > 0:
            band /= band.max()
        band *= 255
        norm_image[:, :, i] = band.astype(np.uint8)

    if export:
        # Update metadata after we're sure norm_image is final
        meta.update({
            'count': norm_image.shape[2],
            'dtype': 'uint8'
        })

        meta.update({'nodata': 0})

        # Transpose back to (C, H, W) for writing
        norm_image_raster = norm_image.transpose(2, 0, 1)

        with rasterio.open(export_path, 'w', **meta) as dst:
            dst.write(norm_image_raster)

    if return_path and export:
        return export_path

    return norm_image


def _to_shapely_params(aff):
    # rasterio Affine(a,b,c,d,e,f) -> shapely (a,b,d,e,c,f)
    return (aff.a, aff.b, aff.d, aff.e, aff.c, aff.f)

def show_in_pixel_space(image, gdf_pixels, idx=None, pad_px=50):

    rgb = None

    with rasterio.open(image) as src:
        gdf_plot = gdf_pixels

        arr = src.read([1,2,3]) if src.count >= 3 else src.read()
        image = np.transpose(arr[:3], (1,2,0)) if arr.ndim==3 else arr
        H, W = image.shape[:2]

        params = _to_shapely_params(src.transform)
        gdf_plot = gdf_pixels.copy()
        gdf_plot["geometry"] = gdf_plot.geometry.apply(lambda g: affine_transform(g, params))
        gdf_plot = gdf_plot.set_crs(src.crs)

        if idx is not None:
            idx = int(gdf_pixels.index[gdf_pixels["id"].eq(idx)][0])
            geom = [gdf_plot.geometry.iloc[idx]]
            out, _ = mask(src, geom, crop=True, nodata=0, filled=True)

            poly = gdf_pixels.geometry.iloc[idx]
            minx, miny, maxx, maxy = poly.bounds  # (x=col, y=row)

            # 2) expand by pad and clamp to image
            col0 = max(0, int(np.floor(minx)) - pad_px)
            col1 = min(W, int(np.ceil(maxx)) + pad_px)
            row0 = max(0, int(np.floor(miny)) - pad_px)
            row1 = min(H, int(np.ceil(maxy)) + pad_px)

            # 3) read only that window (fast; no need to load whole raster)
            win = Window(col0, row0, col1 - col0, row1 - row0)
            arr_2 = src.read([1,2,3], window=win)  # (C, h, w)
            image_2 = np.transpose(arr_2, (1,2,0))  # (h, w, C)

    fig, axes = plt.subplots(1, 3, figsize=(12,12)) if idx is not None else plt.subplots(1, 1, figsize=(8,8))
    ax = axes[0] if idx is not None else axes  
    ax.imshow(image, extent=(0, W, H, 0))  # pixel coords (x:0..W, y:0..H)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)  # y downward to match pixel row

    if idx is None:
        gdf_pixels.boundary.plot(ax=ax, edgecolor="red", linewidth=0.3)
        ax.set_title("All polygons (pixel space)")

    else:
        gdf_pixels.iloc[[idx]].boundary.plot(ax=ax, edgecolor="red", linewidth=1)
        ax.set_title(f"Polygon {idx} Location")

        axes[1].imshow(image_2, extent=(col0, col1, row1, row0), interpolation='nearest')  # keep crisp
        axes[1].set_xlim(col0, col1); axes[1].set_ylim(row1, row0)  # y downward
        axes[1].axis('off')
        axes[1].set_title(f"Polygon {idx} Cropped Context")
        axes[1].axis("off")
        gdf_pixels.iloc[[idx]].boundary.plot(ax=axes[1], edgecolor='red', linewidth=0.3)


        rgb = out[:3] if out.shape[0] >= 3 else out
        rgb = np.transpose(rgb, (1,2,0))
        axes[2].imshow(rgb)
        axes[2].set_title("Polygon view")
        #axes[2].axis('off')

    ax.axis("off")
        
    plt.tight_layout()
    plt.show()

    return rgb


def polygon_overlap_count(image_path, gdf_pixels, touched=True):
    """
    Returns (counts, mask)
      counts[r,c] = number of polygons covering that pixel
      mask[r,c]   = 1 if any polygon covers that pixel, else 0
    """

    with rasterio.open(image_path) as src:
        arr = src.read([1,2,3]) if src.count >= 3 else src.read()
        image = np.transpose(arr[:3], (1,2,0)) if arr.ndim==3 else arr
        H, W = image.shape[:2]

    transform = Affine.identity()

    shapes = [(geom, 1) for geom in gdf_pixels.geometry]

    # Strict (no fattening): all_touched=False means center-in-polygon
    counts = rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=touched,
        dtype="uint16",
        merge_alg=MergeAlg.add      # sum overlaps
    )

    # Binary “has polygon” mask (strict)
    mask = rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=touched,
        dtype="uint8",
        default_value=1             # mark covered pixels as 1
        # merge_alg default (replace) is fine here
    )

    mask_bool = (mask == 1)
    masked_image = image.copy()
    masked_image[mask_bool] = 0

    return counts, mask, mask_bool, ~mask_bool, masked_image