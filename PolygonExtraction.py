#region Imports

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio import features
import shapely
import matplotlib.pyplot as plt
from shapely.affinity import affine_transform
from shapely.geometry import box, Polygon
from skimage import measure, color
from skimage.draw import polygon as draw_polygon
from skimage.feature import graycomatrix, graycoprops
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import gc
import pickle
import time
import warnings
warnings.filterwarnings(
	"ignore", 
	module=r".*sam2.*", 
	message=r".*cannot import name '_C'.*"
)

#endregion


class PolygonExtractor:

	#region Main Functions

	def __init__(self, image_path, mask_size_threshold=10, mask_min_hole_area=10, fire_name=None, pic_number=None, tile_quarter=True):
		self.image_path = image_path
		self.norm_img_path = f"{fire_name}_{pic_number}_norm_img.tif"
		self.mask_size_threshold = mask_size_threshold
		self.mask_min_hole_area = mask_min_hole_area
		self.fire_name = fire_name
		self.pic_number = pic_number
		self.tile_quarter = tile_quarter
		self.idx = 1
		self.eps = 1e-6


	def run_SAM_on_image(self):
		PolygonExtractor.clean_cache()

		with rasterio.open(self.image_path) as src:
			red = src.read(1).astype(np.float32)
			green = src.read(2).astype(np.float32)
			blue = src.read(3).astype(np.float32)
			
			# Normalize bands to 0–1
			self.red = red / (red.max()+self.eps) 
			self.green = green / (green.max()+self.eps)
			self.blue = blue / (blue.max()+self.eps)

			self.img = np.dstack((self.red, self.green, self.blue))  # (H, W, C)
			self.generate_spectral_bands(self.red, self.green, self.blue)
		
		PolygonExtractor.normalize_to_uint8_per_band(self.image_path, export=True, export_path=self.norm_img_path)

		if self.tile_quarter:
			self.tiles = self.__split_image_into_quarters()
		else:
			self.tiles = self.__split_image_into_nine()

		for position in self.tile_windows:
			
			current_image = np.transpose(self.tiles[position]['image'], (1,2,0))

			with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
				
				mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
					model_id="facebook/sam2-hiera-large",
					device="cuda",
					mode="eval",
					hydra_overrides_extra=None,
					apply_postprocessing=False,
					points_per_side=256,
					points_per_batch=64,
					pred_iou_thresh=0.7,
					stability_score_thresh=0.8,
					stability_score_offset=0.7,
					mask_threshold=0.0,
					box_nms_thresh=0.7,
					crop_n_layers=1,
					crop_nms_thresh=0.7,
					crop_overlap_ratio= 512 / 1500,
					crop_n_points_downscale_factor=2,
					point_grids=None,
					min_mask_region_area=self.mask_size_threshold,
					output_mode="binary_mask",
					use_m2m=True,
					multimask_output=False,
				)

				mask_table = mask_generator.generate(current_image)
				
				print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
				mask_generator.predictor.reset_predictor()
				del mask_generator

				PolygonExtractor.clean_cache()
			
			mask_gdf = self.masks_to_shapes(mask_table, position)

			print(f"Position {position} extraction completed, Masks kept: {len(mask_gdf)}/{len(mask_table)}")

			if position == self.tile_windows[0]:
				final_gdf = mask_gdf
			else:
				final_gdf = pd.concat([final_gdf, mask_gdf], ignore_index=True)

			PolygonExtractor.clean_cache()

		final_gdf.attrs["pixel_transform"] = (0,1,0, 0,0,1)	

		print(f"Total polygons before deduplication: {len(final_gdf)}")

		with open(f"{self.fire_name}_{self.pic_number}_polygons.pkl", "wb") as f:
			pickle.dump(final_gdf, f)

		return final_gdf
	
	#endregion

	#region Mask Processing

	def masks_to_shapes(self, mask_dict, tile_location="bottom_right"):
		rows = []    
		
		for i in range(len(mask_dict)):

			seg = mask_dict[i]["segmentation"].astype(np.uint8) 

			if seg.sum() == 0:
				#print(f"Mask {i} is empty, skipping...")
				continue

			if mask_dict[i]["area"] < self.mask_size_threshold:
				#print(f"Mask {i} is too small, skipping...")
				continue

			if mask_dict[i]["bbox"][2] < 5 and mask_dict[i]["bbox"][3] < 5:
				#print(f"Mask {i} bounding box is too small, skipping...")
				continue

			pairs = list(features.shapes(seg, mask=(seg==1)))  # materialize ONCE
			geoms_local = [shapely.geometry.shape(g) for g, v in pairs if v == 1]

			r0, c0 = self.tiles[tile_location]["origin"]
			geoms = [shapely.affinity.translate(g, xoff=c0, yoff=r0) for g in geoms_local]

			if not geoms:
				#print(f"No FG geometry for mask {i}; skipping")
				continue

			# If there are multiple pieces, decide what to do:
			if len(geoms) > 1:
				#print(f"{len(geoms)} geometries for mask {i}; using largest")
				shap = max(geoms, key=lambda g: g.area)
			else:
				shap = geoms[0]

			if(len(shap.interiors) > 0):
				#print(f"Mask {i} has {len(shap.interiors)} holes; using exterior only")
				keep = [ring for ring in shap.interiors if Polygon(ring).area >= self.mask_min_hole_area]
				if keep:
					#print(f"Keeping {len(keep)} holes")
					shap = Polygon(shap.exterior, holes=keep)

			mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=bool)
			x, y = shap.exterior.xy  # x=cols, y=rows from shapely polygon in pixel coords
			rr, cc = draw_polygon(np.array(y), np.array(x), shape=mask.shape)
			mask[rr, cc] = True
			for ring in shap.interiors:
				hx, hy = ring.xy
				rr, cc = draw_polygon(np.array(hy), np.array(hx), shape=mask.shape)
				mask[rr, cc] = False

			masked_img = np.where(mask[..., None], self.img, np.nan)
			
			if np.nanmin(masked_img) == 0 and np.nanmax(masked_img) <= self.eps:
				#print(f"Mask {i} is black, skipping...")
				continue
			
			img_rows, img_cols = np.where(mask)
			props = self.extract_region_properties(mask, shap, img_rows, img_cols)

			if props is None:
				print(f"Skipping mask {i} due to region properties extraction failure")
				continue

			row = {
				"id": self.idx,
				"geometry": shap,
				"class": None,
				"area_px": mask_dict[i]["area"],
				"pred_score": mask_dict[i]["predicted_iou"],
				"stability_score": mask_dict[i]["stability_score"]
			}
		
			final_dict = row | props
			rows.append(final_dict)
			self.idx += 1

		gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)

		return gdf
	
 
	def extract_region_properties(self, mask, shap, rr, cc):
		props = measure.regionprops(mask.astype(np.uint8))

		if len(props) != 1:
			#print(f"Multiple regions found for mask")
			return None
		
		R_mean = np.mean(self.red[rr, cc])
		G_mean = np.mean(self.green[rr, cc])
		B_mean = np.mean(self.blue[rr, cc])

		if R_mean < 0.1 and G_mean < 0.1 and B_mean < 0.1:
			return None
		
		def log_norm(x):
			return np.sign(x) * np.log1p(np.abs(x))
		
		masked_gray = np.where(mask, self.gray, np.nan)
		masked_gray = masked_gray[int(shap.bounds[1]):int(shap.bounds[3])+1, int(shap.bounds[0]):int(shap.bounds[2])+1]

		levels = 64
		valid = np.isfinite(masked_gray)             # inside-polygon pixels
		if valid.sum() < self.mask_size_threshold:
			return None  # too small to be reliable

		vals = masked_gray[valid]
		vmin, vmax = vals.min(), vals.max()
		if vmax - vmin < self.eps:
			return None  # flat patch
		
		q = np.zeros_like(masked_gray, dtype=np.uint8)  # 0 = outside
		q[valid] = np.floor((vals - vmin) / (vmax - vmin + self.eps) * (levels - 1)).astype(np.uint8) + 1

		glcm = graycomatrix(q, distances=(1,2,4,6), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), levels=levels+1, symmetric=True, normed=True)
		
		p = glcm[1:, 1:, :, :].astype(np.float64)        # drop outside (0) bin
		p /= (p.sum(axis=(0,1), keepdims=True) + 1e-12)
		
		contrast = graycoprops(p, 'contrast')
		homogeneity = graycoprops(p, 'homogeneity')
		correlation = graycoprops(p, 'correlation')
		energy = graycoprops(p, 'energy')
		dissimilarity = graycoprops(p, 'dissimilarity')
		ASM = graycoprops(p, 'ASM')
		entropy = graycoprops(p, 'entropy')

		features_contrast = {f'contrast_d{d}': v for d, v in zip((1,2,4,6), contrast.mean(axis=1))}
		features_homogeneity = {f'homogeneity_d{d}': v for d, v in zip((1,2,4,6), homogeneity.mean(axis=1))}
		features_correlation = {f'correlation_d{d}': v for d, v in zip((1,2,4,6), correlation.mean(axis=1))}
		features_energy = {f'energy_d{d}': v for d, v in zip((1,2,4,6), energy.mean(axis=1))}
		features_dissimilarity = {f'dissimilarity_d{d}': v for d, v in zip((1,2,4,6), dissimilarity.mean(axis=1))}
		features_ASM = {f'ASM_d{d}': v for d, v in zip((1,2,4,6), ASM.mean(axis=1))}
		features_entropy = {f'entropy_d{d}': v for d, v in zip((1,2,4,6), entropy.mean(axis=1))}
		
		props = {
				"area": props[0].area,
				"area_bbox": props[0].area_bbox,
				"area_convex": props[0].area_convex,

				"axis_major_length": props[0].major_axis_length,
				"axis_minor_length": props[0].minor_axis_length,
				"axes_aspect_ratio": props[0].axis_major_length / np.clip(props[0].axis_minor_length, a_min=self.eps, a_max=None),
				
				"perimeter_crofton": props[0].perimeter_crofton,
				"circularity": (4 * np.pi * props[0].area) / np.clip(props[0].perimeter_crofton ** 2, a_min=self.eps, a_max=None),
				"eccentricity": props[0].eccentricity,
				"equivalent_diameter_area": props[0].equivalent_diameter_area,
				"extent": props[0].extent,
				"feret_diameter_max": props[0].feret_diameter_max,
				"orientation": props[0].orientation,
				"solidity": props[0].solidity,

				"moments_hu_1": log_norm(props[0].moments_hu[0]),
				"moments_hu_2": log_norm(props[0].moments_hu[1]),
				"moments_hu_3": log_norm(props[0].moments_hu[2]),
				"moments_hu_4": log_norm(props[0].moments_hu[3]),
				"moments_hu_5": log_norm(props[0].moments_hu[4]),
				"moments_hu_6": log_norm(props[0].moments_hu[5]),
				"moments_hu_7": log_norm(props[0].moments_hu[6]),

				"R_mean": R_mean,
				"R_std": np.std(self.red[rr, cc]),
				"G_mean": G_mean,
				"G_std": np.std(self.green[rr, cc]),
				"B_mean": B_mean,
				"B_std": np.std(self.blue[rr, cc]),
				"H_hsv_mean": np.mean(self.H_hsv[rr, cc]),
				"H_hsv_std": np.std(self.H_hsv[rr, cc]),
				"S_hsv_mean": np.mean(self.S_hsv[rr, cc]),
				"S_hsv_std": np.std(self.S_hsv[rr, cc]),
				"V_hsv_mean": np.mean(self.V_hsv[rr, cc]),
				"V_hsv_std": np.std(self.V_hsv[rr, cc]),
				"ExG_mean": np.mean(self.ExG[rr, cc]),
				"ExG_std": np.std(self.ExG[rr, cc]),
				"TGI_mean": np.mean(self.TGI[rr, cc]),
				"TGI_std": np.std(self.TGI[rr, cc]),
				"UBI_mean": np.mean(self.UBI[rr, cc]),
				"UBI_std": np.std(self.UBI[rr, cc]),
				"UBI_hsv_mean": np.mean(self.UBI_hsv[rr, cc]),
				"UBI_hsv_std": np.std(self.UBI_hsv[rr, cc]),
				"L_star_mean": np.mean(self.L_star[rr, cc]),
				"L_star_std": np.std(self.L_star[rr, cc]),
				"a_star_mean": np.mean(self.a_star[rr, cc]),
				"a_star_std": np.std(self.a_star[rr, cc]),
				"b_star_mean": np.mean(self.b_star[rr, cc]),
				"b_star_std": np.std(self.b_star[rr, cc]),
				"gray_mean": np.nanmean(masked_gray),
				"gray_std": np.nanstd(masked_gray),
				"RGND_mean": np.mean(self.RGND[rr, cc]),
				"RGND_std": np.std(self.RGND[rr, cc]),

				"contrast_mean": contrast.mean(),
				"homogeneity_mean": homogeneity.mean(),
				"correlation_mean": correlation.mean(),
				"energy_mean": energy.mean(),
				"dissimilarity_mean": dissimilarity.mean(),
				"ASM_mean": ASM.mean(),
				"entropy_mean": entropy.mean(),    

			}
		
		return props | features_contrast | features_homogeneity | features_correlation | features_energy | features_dissimilarity | features_ASM | features_entropy
    

	def generate_spectral_bands(self, red, green, blue):

		S = red + green + blue + self.eps
		R = red / S; G = green / S; B = blue / S

		self.TGI = -0.5 * (190*(R - G) - 120*(R - B))
		self.RGND = (R - G) / (R + G + self.eps)
		self.ExG  = 2*G - R - B

		hsv = color.rgb2hsv(np.dstack([red,green,blue]))
		self.H_hsv, self.S_hsv, self.V_hsv = hsv[...,0], hsv[...,1], hsv[...,2]

		self.UBI = (R - B) / (R + B + self.eps)
		self.UBI_hsv = (self.H_hsv - self.V_hsv) / (self.H_hsv + self.V_hsv + self.eps)

		lab = color.rgb2lab(np.dstack([red,green,blue]))
		self.L_star, self.a_star, self.b_star = lab[...,0], lab[...,1], lab[...,2]

		self.gray = color.rgb2gray(np.dstack([red,green,blue]))
	

	#endregion

	#region Utilitly Functions	

	def show_gdf_in_pixel_space(self, gdf_pixels, idx=None, pad_px=50):

		def _to_shapely_params(aff):
			# rasterio Affine(a,b,c,d,e,f) -> shapely (a,b,d,e,c,f)
			return (aff.a, aff.b, aff.d, aff.e, aff.c, aff.f)

		with rasterio.open(self.image_path) as src:
			gdf_plot = gdf_pixels

			arr = src.read([1,2,3]) if src.count >= 3 else src.read()
			img = np.transpose(arr[:3], (1,2,0)) if arr.ndim==3 else arr
			H, W = img.shape[:2]

			params = _to_shapely_params(src.transform)
			gdf_plot = gdf_pixels.copy()
			gdf_plot["geometry"] = gdf_plot.geometry.apply(lambda g: affine_transform(g, params))
			gdf_plot = gdf_plot.set_crs(src.crs)

			if idx is not None:
				idx = int(gdf_pixels.index[gdf_pixels["id"].eq(idx)][0])+1
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
				img_2 = np.transpose(arr_2, (1,2,0))  # (h, w, C)

		fig, axes = plt.subplots(1, 3, figsize=(12,12)) if idx is not None else plt.subplots(1, 1, figsize=(8,8))
		ax = axes[0] if idx is not None else axes  
		ax.imshow(img, extent=(0, W, H, 0))  # pixel coords (x:0..W, y:0..H)
		ax.set_xlim(0, W); ax.set_ylim(H, 0)  # y downward to match pixel row

		if idx is None:
			gdf_pixels.boundary.plot(ax=ax, edgecolor="red", linewidth=1)
			ax.set_title("All polygons (pixel space)")
		else:
			gdf_pixels.iloc[[idx]].boundary.plot(ax=ax, edgecolor="red", linewidth=1)
			ax.set_title(f"Polygon {idx} Location")

			axes[1].imshow(img_2, extent=(col0, col1, row1, row0), interpolation='nearest')  # keep crisp
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
			
		plt.tight_layout(); plt.show()

	@staticmethod
	def clean_cache(): 
		if torch.cuda.is_available():
			torch.cuda.synchronize()
			torch.cuda.empty_cache()
			gc.collect()
		

	@staticmethod
	def normalize_to_uint8_per_band(image_path, export=False, export_path="norm_img.tif"):
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

		return norm_image
	
	#endregion
	
	#region Tiling
	
	def __split_image_into_nine(self, overlap_fraction=0.2):

		self.tile_windows = [
            "top_left",
            "top_center",
            "top_right",
            "center_left",
            "center",
            "center_right",
            "bottom_left", 
            "bottom_center",
            "bottom_right",
        ]

		self.tiles = PolygonExtractor.split_image_into_nine(self.image_path, overlap_fraction=overlap_fraction)


	def __split_image_into_quarters(self):

		self.tile_windows = [
            "top_left",
            "top_right",
            "bottom_left", 
            "bottom_right",
        ]

		self.tiles = PolygonExtractor.split_image_into_quarters(self.image_path)
	
	@staticmethod
	def split_image_into_nine(image_path, overlap_fraction=0.2):
		"""
		Splits the image into 9 overlapping tiles with keys indicating relative positions.
		Each tile overlaps its neighbors by a specified fraction.

		Parameters:
			image_path (str): Path to the input image (.tif).
			overlap_fraction (float): Fraction of overlap between adjacent tiles.

		Returns:
			dict: Dictionary of tiles with keys like 'top_left', 'center', etc.
				Each entry contains 'image', 'row_bounds', 'col_bounds', 'transform'.
		"""
		
		with rasterio.open(image_path) as src:
			image = src.read()  # shape: (C, H, W)
			height, width = image.shape[1], image.shape[2]
			
			third_h = height // 3
			third_w = width // 3
			overlap_h = int(third_h * overlap_fraction)
			overlap_w = int(third_w * overlap_fraction)

			# Define windows, row and column bounds
			windows = {
				"top_left":    ((0, third_h+overlap_h),     (0, third_w + overlap_w)),
				"top_center":  ((0, third_h+overlap_h),     (third_w - overlap_w, 2 * third_w + overlap_w)),
				"top_right":   ((0, third_h+overlap_h),     (2 * third_w - overlap_w, width)),
				"center_left": ((third_h-overlap_h, 2 * third_h + overlap_h), (0, third_w + overlap_w)),
				"center":      ((third_h-overlap_h, 2 * third_h + overlap_h), (third_w - overlap_w, 2 * third_w + overlap_w)),
				"center_right":((third_h-overlap_h, 2 * third_h + overlap_h), (2 * third_w - overlap_w, width)),
				"bottom_left": ((2 * third_h - overlap_h, height), (0, third_w + overlap_w)), 
				"bottom_center":((2 * third_h - overlap_h, height), (third_w - overlap_w, 2 * third_w + overlap_w)),
				"bottom_right":((2 * third_h - overlap_h, height), (2 * third_w - overlap_w, width))
			}
			

			tiles = {}
			for key, ((r_start, r_end), (c_start, c_end)) in windows.items():
				tiles[key] = {
					"image": image[:, r_start:r_end, c_start:c_end],
					"origin": (r_start, c_start),  # row, col
					"bounds": box(c_start, r_start, c_end, r_end)
				}


		return tiles
	
	@staticmethod
	def split_image_into_quarters(image_path):
		"""
		Splits the image into 4 tiles

		Parameters:
			image_path (str): Path to the input image (.tif).
			overlap_fraction (float): Fraction of overlap between adjacent tiles.

		Returns:
			dict: Dictionary of tiles with keys like 'top_left', 'bottom_right', etc.
				Each entry contains 'image', 'row_bounds', 'col_bounds', 'transform'.
		"""
		with rasterio.open(image_path) as src:
			image = src.read()  # (C, H, W)

		_, H, W = image.shape
		h_half, w_half = H // 2, W // 2

		# Define windows
		windows = {
			"top_left":    ((0, h_half),     (0, w_half)),
			"top_right":   ((0, h_half),     (w_half, W)),
			"bottom_left": ((h_half, H),     (0, w_half)),
			"bottom_right":((h_half, H),     (w_half, W))
		}

		# Extract quarters
		quarters = {}
		for key, ((r_start, r_end), (c_start, c_end)) in windows.items():
			quarters[key] = {
				"image": image[:, r_start:r_end, c_start:c_end],
				"origin": (r_start, c_start),  # x, y offset
				"bounds": box(c_start, r_start, c_end, r_end)
			}

		return quarters
	
	#endregion
	

#region Main

if __name__ == "__main__":
	fire_name = "santa-rosa"
	pic_number = "0014"

	pre_image_f = f"../fires/{fire_name}/images/{fire_name}-wildfire_0000{pic_number}_pre_disaster.tif"
	post_image_f = f"../fires/{fire_name}/images/{fire_name}-wildfire_0000{pic_number}_post_disaster.tif"
	
	t0 = time.perf_counter()

	extractor = PolygonExtractor(pre_image_f, mask_size_threshold=15, mask_min_hole_area=10, fire_name=fire_name, pic_number=pic_number)
	gdf = extractor.run_SAM_on_image()

	extractor.show_gdf_in_pixel_space(gdf)

	elapsed = time.perf_counter() - t0
	print(f"Elapsed time: {(elapsed/60):.2f} mins")

#endregion