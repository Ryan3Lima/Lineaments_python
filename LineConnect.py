## Imports-------------------------------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import math
import networkx as nx
from shapely.ops import linemerge, unary_union

## Functions -----------------------------------------------------

def angle_between_lines(line1, line2):
    """
        Calculates the absolute angular difference (in degrees) between two lines,
        normalized to 0–180° to avoid directionality.
    """
    def get_angle(line):
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]
        dx = x1 - x0
        dy = y1 - y0
        return math.degrees(math.atan2(dy, dx)) % 180  # modulo 180 to ignore direction

    angle1 = get_angle(line1)
    angle2 = get_angle(line2)
    return abs(angle1 - angle2)

def endpoints_close(line1, line2, dist_thresh=1.0):
    """
        Returns True if any endpoint of line1 is within dist_thresh of any endpoint of line2.
    """
    ends1 = [Point(c) for c in [line1.coords[0], line1.coords[-1]]]
    ends2 = [Point(c) for c in [line2.coords[0], line2.coords[-1]]]
    return any(p1.distance(p2) < dist_thresh for p1 in ends1 for p2 in ends2)

def is_merge_candidate(line1, line2, angle_thresh=10, dist_thresh=2.0):
    angle_diff = angle_between_lines(line1, line2)
    if angle_diff > angle_thresh:
        return False
    return endpoints_close(line1, line2, dist_thresh)

def extend_line(line, distance=2.0):
    from shapely.geometry import LineString
    x0, y0 = line.coords[0]
    x1, y1 = line.coords[-1]

    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    ux = dx / length
    uy = dy / length

    new_start = (x0 - ux * distance, y0 - uy * distance)
    new_end = (x1 + ux * distance, y1 + uy * distance)
    return LineString([new_start, new_end])


if __name__ == "__main__":
    # Create synthetic lines for testing
    lines = [
        LineString([(0, 0), (5, 5)]),  # Line A
        LineString([(5.2, 5.2), (10, 10)]),  # Line B - near A, same direction
        LineString([(0, 5), (5, 0)]),  # Line C - crosses A, different direction
        LineString([(11, 10), (15, 14)]),  # Line D - far, but same angle
        LineString([(6, 7), (6, 11)]),  # Line E - parallel/near B but vertical
        LineString([(5, 7), (5, 11)]),  # Line F - vertical, near A
    ]

    gdf = gpd.GeoDataFrame(geometry=lines)

    # Plot the lines
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(ax=ax, color='black', linewidth=2)

    labels = ['0', '1', '2', '3', '4', '5']  # Labels for the lines

    # Annotate each line with its label
    for idx, line in enumerate(gdf.geometry):
        x, y = line.coords[0]  # Get the starting point of the line
        ax.annotate(labels[idx], (x, y), color='blue', fontsize=12, weight='bold')

    plt.title("Original Line Segments with Annotations")
    plt.axis('equal')
    plt.show()

    # Extend lines for snapping checks only
    extension_dist = 2  # You can tweak this!
    extended_lines = [extend_line(line, distance=extension_dist) for line in gdf.geometry]

    merge_pairs = []

    for i, line1 in enumerate(extended_lines):
        for j, line2 in enumerate(extended_lines):
            if i >= j:
                continue
            if is_merge_candidate(line1, line2, angle_thresh=10, dist_thresh=2.0):
                merge_pairs.append((i, j))

    print("Merge Candidates (by index):", merge_pairs)

    # Base plot: all lines in black
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(ax=ax, color='black', linewidth=2)

    # Highlight merge candidate pairs in red
    for i, j in merge_pairs:
        gdf.iloc[[i, j]].plot(ax=ax, color='red', linewidth=3)

    plt.title("Merge Candidates Highlighted")
    plt.axis('equal')
    plt.show()

    # Build graph from merge pairs
    G = nx.Graph()
    G.add_edges_from(merge_pairs)

    # Each connected component is a group of indices to merge
    groups = list(nx.connected_components(G))
    print("Line groups to merge:", groups)

    merged_lines = []
    merged_indices = set()

    for group in nx.connected_components(nx.Graph(merge_pairs)):
        group = list(group)
        lines_to_merge = [gdf.geometry[i] for i in group]  # original lines
        merged = linemerge(unary_union(lines_to_merge))
        merged_lines.append(merged)
        merged_indices.update(group)

    # Add unmerged lines
    unmerged_lines = [gdf.geometry[i] for i in range(len(gdf)) if i not in merged_indices]
    final_lines = merged_lines + unmerged_lines

    merged_gdf = gpd.GeoDataFrame(geometry=final_lines)

    fig, ax = plt.subplots(figsize=(6, 6))
    merged_gdf.plot(ax=ax, color='blue', linewidth=2)
    plt.title("Extended + Merged Lineaments")
    plt.axis('equal')
    plt.show()


