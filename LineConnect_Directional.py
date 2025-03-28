## Imports-------------------------------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import math
import networkx as nx
from shapely.ops import linemerge, unary_union, nearest_points
#wow
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
        return math.degrees(math.atan2(dy, dx)) % 180
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

def extend_line(line, distance=2.0):
    """
    Extends both ends of a line by 'distance' units in the line's direction.
    """
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

def directional_collinearity_score(line1, line2):
    """
    Computes a collinearity score based on how well line2 continues from line1.
    Lower scores mean better directional collinearity.
    """
    end1 = Point(line1.coords[-1])
    dx = line1.coords[-1][0] - line1.coords[0][0]
    dy = line1.coords[-1][1] - line1.coords[0][1]
    length1 = math.hypot(dx, dy)
    if length1 == 0:
        return float('inf')
    ux, uy = dx / length1, dy / length1
    nearest_on_line2 = nearest_points(end1, line2)[1]
    d = end1.distance(nearest_on_line2)
    projected = Point(end1.x + ux * d, end1.y + uy * d)
    start2 = Point(line2.coords[0])
    return projected.distance(start2)

def is_merge_candidate(line1, line2, angle_thresh=10, dist_thresh=2.0, collinearity_thresh=1.0):
    """
    Returns True only if lines are aligned, endpoints are close, and collinear score is low.
    """
    if angle_between_lines(line1, line2) > angle_thresh:
        return False
    if not endpoints_close(line1, line2, dist_thresh):
        return False
    score = directional_collinearity_score(line1, line2)
    return score < collinearity_thresh

## Main Script -----------------------------------------------------

if __name__ == "__main__":
    # Create synthetic lines for testing
    lines = [
        LineString([(0, 0), (5, 5)]),       # 0
        LineString([(5.2, 5.2), (10, 10)]), # 1
        LineString([(0, 5), (5, 0)]),       # 2
        LineString([(11, 10), (15, 14)]),   # 3
        LineString([(6, 7), (6, 11)]),      # 4
        LineString([(5, 7), (5, 11)]),      # 5
        LineString([(5, 11.5), (6, 15)]),   # 6
    ]
    gdf = gpd.GeoDataFrame(geometry=lines)

    # Plot original lines with labels
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(ax=ax, color='black', linewidth=2)
    for idx, line in enumerate(gdf.geometry):
        x, y = line.coords[0]
        ax.annotate(str(idx), (x, y), color='blue', fontsize=12, weight='bold')
    plt.title("Original Line Segments with Annotations")
    plt.axis('equal')
    plt.show()

    merge_pairs = []
    for i, line1 in enumerate(gdf.geometry):
        for j, line2 in enumerate(gdf.geometry):
            if i >= j:
                continue
            if is_merge_candidate(line1, line2, angle_thresh=10, dist_thresh=2.0, collinearity_thresh=1.0):
                merge_pairs.append((i, j))

    print("Merge Candidates (by index):", merge_pairs)

    # Highlight merge candidate pairs
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(ax=ax, color='black', linewidth=2)
    for i, j in merge_pairs:
        gdf.iloc[[i, j]].plot(ax=ax, color='red', linewidth=3)
    plt.title("Merge Candidates Highlighted")
    plt.axis('equal')
    plt.show()

    # Merge process
    G = nx.Graph()
    G.add_edges_from(merge_pairs)
    groups = list(nx.connected_components(G))
    print("Line groups to merge:", groups)

    merged_lines = []
    merged_indices = set()
    for group in groups:
        lines_to_merge = [gdf.geometry[i] for i in group]
        merged = linemerge(unary_union(lines_to_merge))
        merged_lines.append(merged)
        merged_indices.update(group)

    unmerged_lines = [gdf.geometry[i] for i in range(len(gdf)) if i not in merged_indices]
    final_lines = merged_lines + unmerged_lines

    merged_gdf = gpd.GeoDataFrame(geometry=final_lines)

    fig, ax = plt.subplots(figsize=(6, 6))
    merged_gdf.plot(ax=ax, color='blue', linewidth=2)
    plt.title("Extended + Merged Lineaments (Refined Collinearity)")
    plt.axis('equal')
    plt.show()
