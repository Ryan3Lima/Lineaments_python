import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import LineString, MultiLineString
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def load_and_plot_lineaments(ref_path, pred_path, epsg=26912):
    """
    Load, reproject, clip, and plot two lineament shapefiles.

    Parameters:
        ref_path (str): File path to reference lineaments.
        pred_path (str): File path to predicted lineaments.
        epsg (int): EPSG code for common projected CRS (default: 26912 = NAD83 UTM Zone 12N).
    """
    # Load shapefiles
    ref_lines = gpd.read_file(ref_path)
    pred_lines = gpd.read_file(pred_path)

    # Reproject to common CRS
    ref_lines = ref_lines.to_crs(epsg=epsg)
    pred_lines = pred_lines.to_crs(epsg=epsg)

    # Compute bounds intersection
    ref_bounds = ref_lines.total_bounds
    pred_bounds = pred_lines.total_bounds

    xmin = max(ref_bounds[0], pred_bounds[0])
    ymin = max(ref_bounds[1], pred_bounds[1])
    xmax = min(ref_bounds[2], pred_bounds[2])
    ymax = min(ref_bounds[3], pred_bounds[3])
    intersection_box = box(xmin, ymin, xmax, ymax)

    # Clip to intersection
    ref_clip = gpd.clip(ref_lines, intersection_box)
    pred_clip = gpd.clip(pred_lines, intersection_box)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ref_clip.plot(ax=ax, color='black', linewidth=1, label='Reference')
    pred_clip.plot(ax=ax, color='red', linewidth=0.8, label='Predicted')

    ax.set_title("Lineament Comparison")
    ax.legend()

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add padding to the bottom of the plot
    plt.subplots_adjust(bottom=0.15)

    plt.show()

    return ref_clip, pred_clip


def compare_lineament_lengths(ref_lines, pred_lines):
    """
    Compare total lengths of reference and predicted lineaments.

    Parameters:
        ref_lines (GeoDataFrame): Reference lineaments.
        pred_lines (GeoDataFrame): Predicted lineaments.

    Returns:
        dict: Dictionary with total lengths and ratio.
    """
    ref_lines['length_m'] = ref_lines.geometry.length
    pred_lines['length_m'] = pred_lines.geometry.length

    ref_total = ref_lines['length_m'].sum()
    pred_total = pred_lines['length_m'].sum()
    ratio = pred_total / ref_total if ref_total != 0 else None

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(['Reference', 'Predicted'], [ref_total, pred_total], color=['black', 'red'])
    plt.ylabel('Total Length (meters)')
    plt.title('Total Lineament Length Comparison')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return {
        'ref_total_length_m': ref_total,
        'pred_total_length_m': pred_total,
        'length_ratio': ratio
    }


def safe_line_orientation(geom):
    """Calculate azimuth (0–180°) of a LineString or longest segment of a MultiLineString, handling 2D and 3D."""
    if geom is None or geom.is_empty:
        return np.nan

    # Handle MultiLineString
    if isinstance(geom, MultiLineString):
        if len(geom.geoms) == 0:
            return np.nan
        geom = max(geom.geoms, key=lambda g: g.length)

    # Handle too short geometries
    coords = list(geom.coords)
    if len(coords) < 2:
        return np.nan

    # Handle 2D or 3D coordinates
    x1, y1 = coords[0][:2]
    x2, y2 = coords[-1][:2]

    angle_rad = np.arctan2((y2 - y1), (x2 - x1))
    angle_deg = np.degrees(angle_rad) % 180  # Normalize to 0–180°
    return angle_deg


def plot_orientation_comparison(ref_lines, pred_lines, bins=18):
    """
    Compare orientation of reference and predicted lineaments using
    side-by-side histograms and rose diagrams.

    Parameters:
        ref_lines (GeoDataFrame): Reference lineaments.
        pred_lines (GeoDataFrame): Predicted lineaments.
        bins (int): Number of orientation bins (default 18 for 10° bins).
    """
    # --- Compute orientations ---
    ref_orient = ref_lines.geometry.apply(safe_line_orientation).dropna()
    pred_orient = pred_lines.geometry.apply(safe_line_orientation).dropna()

    # --- Histogram Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].hist(ref_orient, bins=bins, color='black', alpha=0.7)
    axes[1].hist(pred_orient, bins=bins, color='red', alpha=0.7)

    axes[0].set_title('Reference Orientation Histogram')
    axes[1].set_title('Predicted Orientation Histogram')

    for ax in axes:
        ax.set_xlabel('Azimuth (degrees)')
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # --- Rose Diagrams ---
    def rose_data(angles, bins):
        bin_edges = np.linspace(0, 180, bins + 1)
        hist, _ = np.histogram(angles, bins=bin_edges)
        theta = np.deg2rad((bin_edges[:-1] + bin_edges[1:]) / 2)
        return theta, hist

    ref_theta, ref_counts = rose_data(ref_orient, bins)
    pred_theta, pred_counts = rose_data(pred_orient, bins)

    max_count = max(ref_counts.max(), pred_counts.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True), sharey=True)

    for ax, theta, counts, title, color in zip(
            axes,
            [ref_theta, pred_theta],
            [ref_counts, pred_counts],
            ['Reference Rose Diagram', 'Predicted Rose Diagram'],
            ['black', 'red']
    ):
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.bar(theta * 2, counts, width=np.pi / bins, color=color, edgecolor='k', alpha=0.7)
        ax.set_title(title, y=1.1)
        ax.set_ylim(0, max_count)

    plt.tight_layout()
    plt.show()

def compute_buffer_overlap_metrics(ref_lines, pred_lines, buffer_distance=50, plot=True, highlight_fp_fn=True):
    """
    Compute spatial overlap metrics between reference and predicted lineaments using buffer approach.

    Parameters:
        ref_lines (GeoDataFrame): Reference lineaments.
        pred_lines (GeoDataFrame): Predicted lineaments.
        buffer_distance (float): Buffer distance in meters.
        plot (bool): Whether to show the map and confusion matrix.
        highlight_fp_fn (bool): Whether to color FPs and FNs on the map.

    Returns:
        dict: Evaluation metrics including TP/FP/FN lengths and rates.
    """
    # Ensure valid geometries
    ref_lines = ref_lines[ref_lines.is_valid].copy()
    pred_lines = pred_lines[pred_lines.is_valid].copy()

    # Lengths
    ref_lines['length'] = ref_lines.geometry.length
    pred_lines['length'] = pred_lines.geometry.length

    ref_total = ref_lines['length'].sum()
    pred_total = pred_lines['length'].sum()

    # Buffers
    ref_buffer_union = gpd.GeoSeries(ref_lines.buffer(buffer_distance).union_all(), crs=ref_lines.crs)
    pred_buffer_union = gpd.GeoSeries(pred_lines.buffer(buffer_distance).union_all(), crs=pred_lines.crs)

    # Overlaps
    pred_in_buffer = pred_lines[pred_lines.geometry.intersects(ref_buffer_union.iloc[0])]
    ref_in_buffer = ref_lines[ref_lines.geometry.intersects(pred_buffer_union.iloc[0])]

    tp_length = pred_in_buffer['length'].sum()
    fp_length = pred_total - tp_length
    fn_length = ref_total - ref_in_buffer['length'].sum()

    # Percent confusion
    tp_rate = tp_length / ref_total if ref_total else 0
    fp_rate = fp_length / pred_total if pred_total else 0
    fn_rate = fn_length / ref_total if ref_total else 0

    # Metrics
    precision = tp_length / (tp_length + fp_length) if (tp_length + fp_length) else 0
    recall = tp_length / (tp_length + fn_length) if (tp_length + fn_length) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    # Plot
    if plot:
        fig, (ax_map, ax_text) = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [3, 1]})

        # Plot map
        ref_buffer_union.plot(ax=ax_map, color='lightblue', edgecolor='blue', alpha=0.2)

        # Highlight FP/FN if enabled
        if highlight_fp_fn:
            fp_lines = pred_lines[~pred_lines.geometry.intersects(ref_buffer_union.iloc[0])]
            fn_lines = ref_lines[~ref_lines.geometry.intersects(pred_buffer_union.iloc[0])]
            fn_lines.plot(ax=ax_map, color='blue', linewidth=1.2, label='False Negatives')
            fp_lines.plot(ax=ax_map, color='orange', linewidth=1.2, linestyle='--', label='False Positives')

        # Plot base lines
        ref_lines.plot(ax=ax_map, color='black', linewidth=1.0, label='Reference Lines')
        pred_lines.plot(ax=ax_map, color='red', linewidth=0.8,  label='Predicted Lines')
        ax_map.set_title("Buffer Overlap with FP/FN Highlighting")

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='blue', alpha=0.2, label=f'Reference Buffer ({buffer_distance} m)'),
            Line2D([0], [0], color='black', lw=1.2, label='Reference Lines'),
            Line2D([0], [0], color='red', lw=1.2, linestyle='--', label='Predicted Lines'),
            Line2D([0], [0], color='blue', lw=1.5, label='False Negatives'),
            Line2D([0], [0], color='orange', lw=1.5, linestyle='--', label='False Positives')
        ]
        ax_map.legend(handles=legend_elements, loc='upper right')

        # Confusion matrix on right
        ax_text.axis('off')
        matrix_text = (
            f"Confusion Matrix (line length in meters):\n\n"
            f"  TP (correct):       {tp_length:,.1f} m ({tp_rate:.1%} of reference)\n"
            f"  FP (extra):         {fp_length:,.1f} m ({fp_rate:.1%} of predicted)\n"
            f"  FN (missed):        {fn_length:,.1f} m ({fn_rate:.1%} of reference)\n\n"
            f"Metrics:\n"
            f"  Precision:          {precision:.3f}\n"
            f"  Recall:             {recall:.3f}\n"
            f"  F1 Score:           {f1:.3f}\n\n"
            f"Totals:\n"
            f"  Predicted Length:   {pred_total:,.1f} m\n"
            f"  Reference Length:   {ref_total:,.1f} m"
        )
        ax_text.text(0, 1, matrix_text, va='top', ha='left', fontsize=10, family='monospace')

        plt.tight_layout()
        plt.show()

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp_length': tp_length,
        'fp_length': fp_length,
        'fn_length': fn_length,
        'tp_rate': tp_rate,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'pred_total_length': pred_total,
        'ref_total_length': ref_total
    }

def compare_lineament_density(ref_lines, pred_lines, cell_size=1000, plot=True):
    """
    Compare lineament density in grid cells between reference and predicted datasets.

    Parameters:
        ref_lines (GeoDataFrame): Reference lineaments (projected).
        pred_lines (GeoDataFrame): Predicted lineaments (projected).
        cell_size (float): Grid cell size in map units (meters).
        plot (bool): Whether to show density maps and correlation.

    Returns:
        DataFrame: Grid with density values and comparison stats.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd

    # Get intersection bounds
    xmin, ymin, xmax, ymax = ref_lines.total_bounds
    xmin2, ymin2, xmax2, ymax2 = pred_lines.total_bounds
    xmin = max(xmin, xmin2)
    xmax = min(xmax, xmax2)
    ymin = max(ymin, ymin2)
    ymax = min(ymax, ymax2)

    # Create grid cells
    cols = list(np.arange(xmin, xmax, cell_size))
    rows = list(np.arange(ymin, ymax, cell_size))
    cells = []
    for x in cols:
        for y in rows:
            cells.append(box(x, y, x + cell_size, y + cell_size))
    grid = gpd.GeoDataFrame({'geometry': cells}, crs=ref_lines.crs)
    grid['cell_id'] = range(len(grid))
    cell_area_km2 = (cell_size / 1000)**2

    # Helper: compute length in each grid cell
    def calc_density(lines, grid):
        length_list = []
        for cell in grid.geometry:
            clipped = lines[lines.intersects(cell)].copy()
            if not clipped.empty:
                clipped = clipped.intersection(cell)
                length = clipped.length.sum()
            else:
                length = 0
            length_list.append(length / 1000)  # meters → km
        return np.array(length_list) / cell_area_km2  # km / km²

    grid['ref_density'] = calc_density(ref_lines, grid)
    grid['pred_density'] = calc_density(pred_lines, grid)

    # Compute comparison stats
    rmse = np.sqrt(mean_squared_error(grid['ref_density'], grid['pred_density']))
    corr, _ = pearsonr(grid['ref_density'], grid['pred_density'])

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        grid.plot(column='ref_density', cmap='viridis', legend=True, ax=axes[0])
        axes[0].set_title('Reference Lineament Density (km/km²)')

        grid.plot(column='pred_density', cmap='viridis', legend=True, ax=axes[1])
        axes[1].set_title('Predicted Lineament Density (km/km²)')

        plt.suptitle(f"Lineament Density Comparison\nRMSE = {rmse:.3f}, Pearson r = {corr:.3f}", fontsize=14)
        plt.tight_layout()
        plt.show()

    return grid[['geometry', 'ref_density', 'pred_density']], {'rmse': rmse, 'pearson_r': corr}

if __name__ == "__main__":
    REF_LINES_PATH = r'G:\Lineaments_python\Faults_LM1_1m_ExportFeatures.shp'
    PRED_LINES_PATH = r'G:\Lineaments_python\LM1_1mDEM_r_mdhs_B1_defaultLINE.shp'

    ref_lines, pred_lines = load_and_plot_lineaments(REF_LINES_PATH, PRED_LINES_PATH)

    length_stats = compare_lineament_lengths(ref_lines, pred_lines)
    print(length_stats)

    plot_orientation_comparison(ref_lines, pred_lines)

    # metrics = compute_buffer_overlap_metrics(ref_lines, pred_lines, buffer_distance=50)
    metrics = compute_buffer_overlap_metrics(ref_lines, pred_lines, buffer_distance=50, highlight_fp_fn=False)

    grid_df, density_stats = compare_lineament_density(ref_lines, pred_lines, cell_size=30)
    print(density_stats)





