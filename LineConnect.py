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


def directional_collinearity_score(line1, line2):
    """
    Computes a 'collinearity score' based on how well line2 continues from line1.
    Lower scores mean better directional collinearity.
    """
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    # Get end of line1
    end1 = Point(line1.coords[-1])

    # Direction vector of line1
    dx = line1.coords[-1][0] - line1.coords[0][0]
    dy = line1.coords[-1][1] - line1.coords[0][1]
    length1 = math.hypot(dx, dy)
    if length1 == 0:
        return float('inf')  # degenerate line

    ux, uy = dx / length1, dy / length1

    # Find closest point on line2 to end of line1
    nearest_on_line2 = nearest_points(end1, line2)[1]
    d = end1.distance(nearest_on_line2)

    # Project line1 forward by d
    projected = Point(end1.x + ux * d, end1.y + uy * d)

    # Measure how close this projected point is to the *start* of line2
    start2 = Point(line2.coords[0])
    return projected.distance(start2)

def is_merge_candidate(line1, line2, angle_thresh=10, dist_thresh=2.0, collinearity_thresh=1.0):
    """
    Uses angle, endpoint proximity, and directional collinearity score.
    """
    if angle_between_lines(line1, line2) > angle_thresh:
        return False
    if not endpoints_close(line1, line2, dist_thresh):
        return False
    score = directional_collinearity_score(line1, line2)
    return score < collinearity_thresh

def extend_line(line, distance=2.0): # do we need this function anymore?
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

def find_merge_pairs(gdf, angle_thresh=16, dist_thresh=2.0, collinearity_thresh=0.5):
    merge_pairs = []
    for i, line1 in enumerate(gdf.geometry):
        for j, line2 in enumerate(gdf.geometry):
            if i >= j:
                continue
            if is_merge_candidate(line1, line2, angle_thresh, dist_thresh, collinearity_thresh):
                merge_pairs.append((i, j))
    return merge_pairs

def connect_lines_by_bridge(lines_to_merge):
    endpoints = []
    for ln in lines_to_merge:
        endpoints.append(Point(ln.coords[0]))
        endpoints.append(Point(ln.coords[-1]))

    min_dist = float("inf")
    pair = (None, None)
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            d = endpoints[i].distance(endpoints[j])
            if d < min_dist:
                min_dist = d
                pair = (endpoints[i], endpoints[j])

    connector = LineString([pair[0], pair[1]])
    combined = unary_union(lines_to_merge + [connector])
    return linemerge(combined), connector

def locally_merge_lines(lines, search_dist=10.0, angle_thresh=3, collinearity_thresh=0.5):
    lines = list(lines)  # mutable copy
    i = 0
    while i < len(lines):
        line = lines[i]
        match_found = False

        for j in range(len(lines)):
            if i == j:
                continue

            # Skip if lines are far apart
            if line.distance(lines[j]) > search_dist:
                continue

            # Check angle, collinearity, etc.
            if is_merge_candidate(line, lines[j], angle_thresh, dist_thresh=search_dist, collinearity_thresh=collinearity_thresh):
                # Build bridge + merged line
                merged_line, connector = connect_lines_by_bridge([line, lines[j]])

                # Replace line i with merged
                lines[i] = merged_line

                # Remove line j
                lines.pop(j)
                match_found = True
                break  # restart with updated line[i]

        if not match_found:
            i += 1  # move to next line

    return lines

if __name__ == "__main__":
    # Create synthetic lines for testing
    lines = [
        LineString([(0, 0), (5, 5)]),  # Line 0
        LineString([(5.2, 5.2), (10, 10)]),  # Line 1- near 0, same direction
        LineString([(0, 5), (5, 0)]),  # Line 2 - crosses 0, different direction
        LineString([(11, 10), (15, 14)]),  # Line 3 - far, but same angle
        LineString([(6, 7), (6, 11)]),  # Line 4 - parallel/near 1 but vertical
        LineString([(5, 7), (5, 11)]),  # Line 5 - vertical, near 1
        LineString([(5, 11.5), (6, 15)]),  # Line 6 -
        LineString([(10.2, 10.2), (12.2, 12.2)]),  # Line 7
        LineString([(12.5, 12.5), (15.4, 15.2)]),  # Line 8
        LineString([(6, 12.3), (7, 15.4)]),  # Line 9
        LineString([(6, 0), (12, 1)]),  # Line 10
    ]

    gdf = gpd.GeoDataFrame(geometry=lines)

    # Plot the lines
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(ax=ax, color='black', linewidth=2)

    labels = ['0', '1', '2', '3', '4', '5','6','7','8','9','10']  # Labels for the lines


    # Annotate each line with its label
    for idx, line in enumerate(gdf.geometry):
        x, y = line.coords[0]  # Get the starting point of the line
        ax.annotate(labels[idx], (x, y), color='blue', fontsize=12, weight='bold')

    plt.title("Original Line Segments with Annotations")
    plt.axis('equal')
    plt.show()

    new_merged_lines = locally_merge_lines(lines, search_dist=10.0, angle_thresh=20, collinearity_thresh=0.5)
    gdf_new = gpd.GeoDataFrame(geometry=new_merged_lines)

    # Plot the lines
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf_new.plot(ax=ax, color='blue', linewidth=2)
    gdf.plot(ax=ax, color='red', linewidth=2)

    # Annotate each line with its label
    #for idx, line in enumerate(gdf_new.geometry):
    #   x, y = line.coords[0]  # Get the starting point of the line
        # ax.annotate(labels[idx], (x, y), color='blue', fontsize=12, weight='bold')

    plt.title("Merged Lines")
    plt.axis('equal')
    plt.show()



