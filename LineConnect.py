## Imports-------------------------------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.pyplot import annotate
from shapely.geometry import LineString, Point
import math
from shapely.ops import linemerge, unary_union
import logging

## Functions -----------------------------------------------------

def plot_lines(gdf, annotate = False, ax=None):
    # Plot the lines
    fig, ax = plt.subplots(figsize=(6, 6))
    gdf.plot(ax=ax, color='black', linewidth=2)

    if annotate:
        # Annotate each line with its label
        for idx, line in enumerate(gdf.geometry):
            x, y = line.coords[0]  # Get the starting point of the line
            ax.annotate(str(idx), (x+.1, y+.1), color='blue', fontsize=10, weight='bold')
    plt.title("Original Line Segments with Annotations")
    plt.axis('equal')
    plt.show()

def flatten_multilines(gdf):
    """
    Converts MultiLineStrings into separate LineStrings,
    and assigns a 'parent_id' to track origin.
    """
    from shapely.geometry import MultiLineString

    single_parts = []
    parent_ids = []
    for idx, geom in enumerate(gdf.geometry):
        if geom.geom_type == 'LineString':
            single_parts.append(geom)
            parent_ids.append(idx)
        elif geom.geom_type == 'MultiLineString':
            for part in geom.geoms:
                single_parts.append(part)
                parent_ids.append(idx)
        else:
            print(f"Unsupported geometry type: {geom.geom_type}")
    return gpd.GeoDataFrame({'geometry': single_parts, 'parent_id': parent_ids}, crs=gdf.crs)


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


def connect_lines_by_bridge(lines_to_merge):
    """
    Connects the nearest endpoints of a set of lines using a straight bridge line.
    Returns a merged LineString (or longest part of MultiLineString) and the connector.
    """
    endpoints = []
    for ln in lines_to_merge:
        endpoints.append(Point(ln.coords[0]))
        endpoints.append(Point(ln.coords[-1]))

    # Find closest pair of endpoints
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
    merged = linemerge(combined)

    # Handle case where linemerge returns MultiLineString
    if merged.geom_type == 'MultiLineString':
        # Option 1: pick longest part
        merged = max(merged.geoms, key=lambda g: g.length)

    return merged, connector



def locally_merge_lines(gdf, search_dist=10.0, angle_thresh=20, collinearity_thresh=0.5):
    """
        Greedily merges nearby lines, avoiding merging parts of the same parent feature.
        Returns merged lines, connector lines, and the number of merges performed.
    """
    lines = list(gdf.geometry)
    # Assign parent IDs if not already present
    if 'parent_id' in gdf.columns:
        parent_ids = list(gdf['parent_id'])
    else:
        parent_ids = list(range(len(lines)))  # assign unique ID to each line

    connectors = []
    merge_count = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        match_found = False
        # Find candidate indices within range
        candidates = []
        for j in range(len(lines)):
            if i == j:
                continue
            if parent_ids[i] == parent_ids[j]:
                continue
            d = line.distance(lines[j])
            if d <= search_dist:
                candidates.append((j, d))

        # Sort by distance (closest first)
        candidates = sorted(candidates, key=lambda x: x[1])

        # Now iterate through sorted candidates
        for j, _ in candidates:
            if is_merge_candidate(line, lines[j], angle_thresh, search_dist, collinearity_thresh):
                merged_line, connector = connect_lines_by_bridge([line, lines[j]])
                lines[i] = merged_line
                parent_ids[i] = min(parent_ids[i], parent_ids[j])
                lines.pop(j)
                parent_ids.pop(j)
                connectors.append(connector)
                merge_count += 1
                match_found = True
                break

        if not match_found:
            i += 1

    print(f'{merge_count} merges were completed')
    return lines, connectors


def plot_results(original_gdf, merged_gdf, connectors_gdf, search_dist, angle_thresh, collinearity_thresh, annotate = False):
    fig, ax = plt.subplots(figsize=(8, 8))
    merged_gdf.plot(ax=ax, color='blue', linewidth=2, label='Merged Lines')
    original_gdf.plot(ax=ax, color='red', linewidth=2, label='Original Lines')
    if not connectors_gdf.empty:
        connectors_gdf.plot(ax=ax, color='green', linewidth=2, linestyle=':', label='Connectors')
    if annotate:
        for idx, line in enumerate(original_gdf.geometry):
            x, y = line.coords[0]  # Get the starting point of the line
            ax.annotate(str(idx), (x+.1, y+.1), color='blue', fontsize=10, weight='bold')
    else:
        print('Lines not annotated')
    subtitle = f"search_dist: {search_dist}, angle_thresh: {angle_thresh}°, collinearity_thresh: {collinearity_thresh}"
    plt.title("Merged Lineaments with Connectors\n" + subtitle)
    plt.legend()
    plt.axis('equal')
    plt.show()

def report_metadata(gdf_original, gdf_new, search_dist, angle_thresh, collinearity_thresh):
    print("\n--- Lineament Merge Report ---")
    print("Original GeoDataFrame:")
    print(f" - CRS: {gdf_original.crs}")
    print(f" - Feature count: {len(gdf_original)}")
    print("New GeoDataFrame:")
    print(f" - CRS: {gdf_new.crs}")
    print(f" - Feature count: {len(gdf_new)}")
    print(f"Parameters used:")
    print(f" - Search distance: {search_dist} {gdf_original.crs.axis_info[0].unit_name if gdf_original.crs else 'units'}")
    print(f" - Angle threshold: {angle_thresh} degrees")
    print(f" - Collinearity threshold: {collinearity_thresh} {gdf_original.crs.axis_info[0].unit_name if gdf_original.crs else 'units'}")
    print("-------------------------------\n")

if __name__ == "__main__":
    #Create synthetic lines for testing
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
    plot_lines(gdf, annotate=True)

    search_dist = 5
    angle_thresh = 20
    collinearity_thresh = 1


    new_lines, connectors = locally_merge_lines(gdf, search_dist, angle_thresh, collinearity_thresh)
    gdf_new = gpd.GeoDataFrame(geometry=new_lines, crs=gdf.crs)
    connectors_gdf = gpd.GeoDataFrame(geometry=connectors, crs=gdf.crs)

    # Save connectors to file
    #connectors_gdf.to_file("connectors_only.shp")
    #gdf_new.to_file("Merged_lines.shp")

    logging.basicConfig(filename='merge_log.txt', level=logging.INFO)

    logging.info("Started processing")
    logging.info(f"Input CRS: {gdf.crs}")
    logging.info(f"Number of features: {len(gdf)}")

    # Plot
    plot_results(gdf, gdf_new, connectors_gdf, search_dist, angle_thresh, collinearity_thresh)

    # Metadata Report
    report_metadata(gdf, gdf_new, search_dist, angle_thresh, collinearity_thresh)

    # input_path = "Lineament_sample.shp"  # replace with your shapefile path
    # gdf = gpd.read_file(input_path)
    # gdf = flatten_multilines(gdf)
    #
    # plot_lines(gdf)
    #
    # search_dist = 1000
    # angle_thresh = 40
    # collinearity_thresh = 10
    #
    # new_lines, connectors = locally_merge_lines(gdf, search_dist, angle_thresh, collinearity_thresh)
    # gdf_new = gpd.GeoDataFrame(geometry=new_lines, crs=gdf.crs)
    # connectors_gdf = gpd.GeoDataFrame(geometry=connectors, crs=gdf.crs)

    # Save connectors to file
    # connectors_gdf.to_file("connectors_only.shp")
    # gdf_new.to_file("Merged_lines.shp")

    logging.basicConfig(filename='merge_log.txt', level=logging.INFO)

    logging.info("Started processing")
    logging.info(f"Input CRS: {gdf.crs}")
    logging.info(f"Number of features: {len(gdf)}")

    # Plot
    plot_results(gdf, gdf_new, connectors_gdf, search_dist, angle_thresh, collinearity_thresh)

    # Metadata Report
    report_metadata(gdf, gdf_new, search_dist, angle_thresh, collinearity_thresh)



