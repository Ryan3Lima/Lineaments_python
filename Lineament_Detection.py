import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider
import os
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import uniform_filter
from skimage.transform import probabilistic_hough_line
from shapely.geometry import LineString, shape
import geopandas as gpd
import rasterio
from rasterio.features import shapes, rasterize
from skimage.draw import line
import math
from pyproj import CRS



DEM_PATH = r'sample_data/DEM.tif'
SHAPEFILE_PATH =r'sample_data/Manual_Lineaments_LM.shp'

def rasterize_shapefile(shapefile_path, reference_raster_path):
    """
    Rasterizes a shapefile based on a reference raster.

    Returns: mask (np.array), gdf (GeoDataFrame), crs (CRS), transform (Affine)
    """
    gdf = gpd.read_file(shapefile_path)

    with rasterio.open(reference_raster_path) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    return mask, gdf, crs, transform

def load_dem_clean(dem_path, handle_nodata=True):
    """
    Loads a DEM and returns a masked array if handle_nodata is True.

    Returns:
        dem (np.ndarray): Cleaned DEM data
        profile (dict): Raster profile (metadata)
        transform (Affine): Affine transform
        crs (CRS): Coordinate reference system
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if handle_nodata:
        mask = np.isnan(dem)
        if nodata is not None:
            mask |= (dem == nodata)
        dem = np.ma.masked_array(dem, mask=mask)
        print("Masked pixels:", np.sum(dem.mask))
        print("DEM min/max (valid):", dem.min(), dem.max())

    return dem, profile, transform, crs

def plot_lines_over_dem(dem_path, shapefile_path, cmap='terrain', figsize=(12, 8),
                        lin_color='red', lin_width=0.5):
    """
    Load and plot a shapefile of lines on top of a cleaned DEM.

    Parameters:
        dem_path (str): Path to DEM raster
        shapefile_path (str): Path to shapefile with line features
        cmap (str): Colormap for DEM
        figsize (tuple): Size of matplotlib figure
        lin_color (str): Color for linework
        lin_width (float): Line width
    """

    # Load DEM with nodata handling enabled
    dem, profile, transform, dem_crs = load_dem_clean(dem_path, handle_nodata=True)
    print(f"DEM CRS = {dem_crs}")

    # Calculate extent from raster transform
    extent = [
        transform[2],
        transform[2] + dem.shape[1] * transform[0],
        transform[5] + dem.shape[0] * transform[4],
        transform[5]
    ]

    # Load shapefile and reproject if needed
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != dem_crs:
        print("⚠️ Reprojecting shapefile to match DEM CRS.")
        gdf = gdf.to_crs(dem_crs)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    dem_show = ax.imshow(dem, extent=extent, cmap=cmap, origin='upper')  # masked array will be respected

    gdf.plot(ax=ax, edgecolor=lin_color, linewidth=lin_width)

    # Set aspect ratio
    if dem_crs.is_geographic:
        mean_lat = (extent[2] + extent[3]) / 2
        ax.set_aspect(1 / np.cos(np.deg2rad(mean_lat)))
    else:
        ax.set_aspect('equal')

    # Title and clean look
    basename = os.path.splitext(os.path.basename(shapefile_path))[0]
    ax.set_title(f"{basename} over DEM")
    ax.axis('off')
    plt.colorbar(dem_show, ax=ax, fraction=0.03)
    plt.show()



def plot_canny_edges(dem_path, threshold1=1.0, threshold2=10.0, save_lines=False, debug=False):

    dem, profile, transform, crs = load_dem_clean(dem_path)

    # Normalize DEM to 8-bit grayscale
    dem_min, dem_max = np.min(dem), np.max(dem)
    #dem_norm = ((dem - dem_min) / (dem_max - dem_min) * 255).astype(np.uint8)
    dem_norm = ((dem - dem_min) / (dem_max - dem_min) * 255).filled(0).astype(np.uint8)
    # .filled(0) replaces masked values with 0 just for the normalization and conversion step, avoiding the runtime warning.
    if debug:
        print("DEM normalized successfully. Range:", dem_norm.min(), dem_norm.max())
    # Gaussian blur for noise reduction
    dem_blurred = cv2.GaussianBlur(dem_norm, (3, 3), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(dem_blurred, threshold1, threshold2)
    # Extract and plot contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fig, ax = plt.subplots(figsize=(12, 8))
    dem_show = ax.imshow(dem, cmap='terrain', origin='upper')

    line_geoms, lengths, angles = [], [], []
    for cnt in contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) == 2:
            coords = [~transform * tuple(pt) for pt in cnt]
            line = LineString(coords)
            if line.is_valid and line.length > 1:
                ax.plot(cnt[:, 0], cnt[:, 1], color='red', linewidth=0.5)
                line_geoms.append(line)
                lengths.append(line.length)
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                angle = (math.degrees(math.atan2(dy, dx)) + 360) % 180
                angles.append(angle)

    ax.set_title(f"Canny Edges (T1={threshold1}, T2={threshold2})")
    ax.axis('off')
    plt.colorbar(dem_show, ax=ax, fraction=0.03)
    plt.show()

    if save_lines and line_geoms:
        output_dir = os.path.join(os.getcwd(), 'out_canny')
        os.makedirs(output_dir, exist_ok=True)

        lines_gdf = gpd.GeoDataFrame({
            'length_m': lengths,
            'angle_deg': angles,
            'geometry': line_geoms
        }, crs=crs)

        basename = os.path.splitext(os.path.basename(dem_path))[0]
        filename = f"canny_lines_{basename}_T1_{threshold1}_T2_{threshold2}.shp"
        output_path = os.path.join(output_dir, filename)
        lines_gdf.to_file(output_path)
        print(f"✅ Saved {len(lines_gdf)} lines to: {output_path}")



def plot_local_flood_edges(dem_path, window_size=11, offset=5.0,
                           save_lines=False, debug=False, save_rose=False):


    # Load the DEM
    dem, profile, transform, crs = load_dem_clean(dem_path)

    # Local mean elevation (moving window average)
    dem_filled = dem.filled(np.nan) if np.ma.isMaskedArray(dem) else dem.astype(float)

    # Replace NaNs temporarily with a neutral value (like the mean or median of DEM)
    neutral_val = np.nanmedian(dem_filled)
    dem_safe = np.where(np.isnan(dem_filled), neutral_val, dem_filled)

    local_mean = uniform_filter(dem_safe, size=window_size, mode='nearest')
    # delta = dem - local_mean #DEBUG
    # print("Δ Elevation (DEM - Local Mean):") #DEBUG
    # print("  Min:", np.min(delta)) #DEBUG
    # print("  Max:", np.max(delta)) #DEBUG
    # print("  Mean:", np.mean(delta)) #DEBUG

    # Flood if pixel is lower than local mean - offset
    #flooded = (dem <= (local_mean - offset)).astype(np.uint8)
    flooded = (dem <= (local_mean - offset)).astype(np.uint8)
    flooded_count = np.sum(flooded)

    if debug:
        print(f"Flooded pixels: {flooded_count}")

    if flooded_count == 0:
        print("⚠️ No pixels meet flood condition — try reducing the offset or window size.")
        return
    # plt.figure(figsize=(8, 6)) #DEBUG
    # plt.imshow(flooded, cmap='gray') #DEBUG
    # plt.title("Flooded Mask (1 = flooded)") #DEBUG
    # plt.show() #DEBUG
    # Skeletonize flooded regions

    skeleton = skeletonize(flooded)

    # Label connected features
    labeled = label(skeleton)
    props = regionprops(labeled)

    # Prepare figure for DEM + rose diagram layout
    fig, ax = plt.subplots(figsize=(12, 8))
    dem_show = ax.imshow(dem, cmap='terrain', origin='upper')
    ax.contour(skeleton, colors='red', linewidths=0.5)

    line_geoms, lengths, angles = [], [], []

    for p in props:
        if p.major_axis_length > 20 and p.eccentricity > 0.9:
            y0, x0 = p.centroid
            ax.plot(x0, y0, 'bo', markersize=2)

            length = p.major_axis_length
            orientation = p.orientation

            x1 = x0 + np.cos(orientation) * 0.5 * length
            y1 = y0 - np.sin(orientation) * 0.5 * length
            x2 = x0 - np.cos(orientation) * 0.5 * length
            y2 = y0 + np.sin(orientation) * 0.5 * length

            pt1 = ~transform * (x1, y1)
            pt2 = ~transform * (x2, y2)
            line = LineString([pt1, pt2])

            if line.is_valid and line.length > 0:
                line_geoms.append(line)
                lengths.append(line.length)
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                angle = (math.degrees(math.atan2(dy, dx)) + 360) % 180
                angles.append(angle)

    ax.set_title(f"Local Flooding (window={window_size}×{window_size}, offset={offset:.1f}m)")
    ax.axis('off')
    plt.colorbar(dem_show, ax=ax, fraction=0.03)
    plt.show()

    if save_lines and line_geoms:
        output_dir = os.path.join(os.getcwd(), 'out_flood')
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(dem_path))[0]
        filename = f"flood_lines_{basename}_win{window_size}_off{offset:.1f}.shp"
        output_path = os.path.join(output_dir, filename)

        gdf = gpd.GeoDataFrame({
            'length_m': lengths,
            'angle_deg': angles,
            'geometry': line_geoms
        }, crs=crs)

        gdf.to_file(output_path)
        print(f"✅ Saved {len(gdf)} lines to: {output_path}")
    if debug:
        print("DEM shape:", dem.shape)
        print("Masked pixels:", np.sum(np.ma.getmaskarray(dem)))
        print("Region count:", len(props))


def run_hough_p(mask, gdf, spatial_ref, transform,
                rho=2, theta_deg=0.5, threshold=30,
                min_line_length=10, max_line_gap=4,
                save_lines=False, save_rose=False):

    output_dir = os.getcwd()
    theta = np.deg2rad(theta_deg)

    # Ensure mask is binary 0/255
    edges = (mask > 0).astype(np.uint8) * 255

    # Run Hough Transform
    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    line_geoms, lengths, angles = [], [], []
    count = 0

    # Plot setup
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 12),
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.imshow(edges, cmap='gray', origin='upper')
    gdf.plot(ax=ax1, edgecolor='cyan', linewidth=0.5)

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            ax1.plot([x1, x2], [y1, y2], color='red', linewidth=1)
            count += 1

            # Convert pixel to geographic coordinates
            pt1 = ~transform * (x1, y1)
            pt2 = ~transform * (x2, y2)
            line_geom = LineString([pt1, pt2])

            if line_geom.is_valid and line_geom.length > 0:
                line_geoms.append(line_geom)
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                length = math.hypot(dx, dy)
                angle = (math.degrees(math.atan2(dy, dx)) + 360) % 180  # [0, 180)

                lengths.append(length)
                angles.append(angle)

    ax1.set_title(f"HoughLinesP: {count} segments (ρ={rho}, θ={theta_deg:.2f}°, minLen={min_line_length}, gap={max_line_gap})")
    ax1.axis('off')

    # --- Rose diagram ---
    if angles:
        bins = np.linspace(0, 180, 19)
        counts, _ = np.histogram(angles, bins=bins)
        theta_centers = np.deg2rad((bins[:-1] + bins[1:]) / 2)
        width = np.deg2rad(10)

        ax2.clear()
        ax2 = plt.subplot(212, polar=True)
        ax2.bar(theta_centers, counts, width=width, bottom=0.0, align='center', edgecolor='black')
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_title("Orientation Rose Diagram", va='bottom')

        if save_rose:
            rose_path = os.path.join(output_dir, f"rose_rho{rho}_theta{theta_deg:.2f}_minlen{min_line_length}.png")
            plt.savefig(rose_path, dpi=300, bbox_inches='tight')
            print(f"📸 Rose diagram saved to: {rose_path}")

    plt.tight_layout()
    plt.show()

    # --- Save shapefile ---
    if save_lines and line_geoms:
        spatial_crs = CRS.from_user_input(spatial_ref)
        lines_gdf = gpd.GeoDataFrame({
            'length_m': lengths,
            'angle_deg': angles,
            'geometry': line_geoms
        }, crs=spatial_crs.to_wkt())

        filename = f"hough_lines_rho{rho}_theta{theta_deg:.2f}_thresh{threshold}_minlen{min_line_length}.shp"
        output_path = os.path.join(output_dir, filename)
        lines_gdf.to_file(output_path)
        print(f"✅ Saved {len(lines_gdf)} lines to: {output_path}")
    elif not save_lines:
        print("ℹ️ Export skipped by user.")
    else:
        print("⚠️ No lines detected.")


#---------------------------------------------

if __name__ == "__main__":
    # Use float sliders for finer control

    if  os.path.exists(DEM_PATH):
        print(f'DEM_PATH = {DEM_PATH} exists!!!')

    else:
        print(f'DEM_PATH = {DEM_PATH} DOES NOT EXIST!!!')

    if  os.path.exists(SHAPEFILE_PATH):
        print(f'SHAPEFILE_PATH = {SHAPEFILE_PATH} exists!!!')

    else:
        print(f'SHAPEFILE_PATH = {SHAPEFILE_PATH} DOES NOT EXIST!!!')

    plot_lines_over_dem(dem_path=DEM_PATH,shapefile_path=SHAPEFILE_PATH)

    plot_local_flood_edges(
        dem_path="sample_data/DEM.tif",
        window_size=11,
        offset=1.2,
        debug=True
    )

    shapefile_path = r'out_canny/shapefile2/canny_lines_DEM_T1_0.4_T2_34.0.shp'
    reference_raster_path = DEM_PATH

    mask, gdf, crs, transform = rasterize_shapefile(shapefile_path, reference_raster_path)

    run_hough_p(mask, gdf, crs, transform,
                rho=2, theta_deg=0.5, threshold=30,
                min_line_length=10, max_line_gap=4,
                save_lines=True, save_rose=True)




