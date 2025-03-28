# def connect_lines_by_bridge(lines_to_merge):
#     """
#        Connects the nearest endpoints of a set of lines using a straight bridge line.
#        Returns a merged LineString (or longest part of MultiLineString) and the connector.
#     """
#     endpoints = []
#     for ln in lines_to_merge:
#         endpoints.append(Point(ln.coords[0]))
#         endpoints.append(Point(ln.coords[-1]))
#
#     min_dist = float("inf")
#     pair = (None, None)
#     for i in range(len(endpoints)):
#         for j in range(i+1, len(endpoints)):
#             d = endpoints[i].distance(endpoints[j])
#             if d < min_dist:
#                 min_dist = d
#                 pair = (endpoints[i], endpoints[j])
#
#     connector = LineString([pair[0], pair[1]])
#     combined = unary_union(lines_to_merge + [connector])
#     return linemerge(combined), connector


# def locally_merge_lines(lines, search_dist=10.0, angle_thresh=3, collinearity_thresh=0.5):
#     lines = list(lines)  # mutable copy
#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         match_found = False
#
#         for j in range(len(lines)):
#             if i == j:
#                 continue
#
#             # Skip if lines are far apart
#             if line.distance(lines[j]) > search_dist:
#                 continue
#
#             # Check angle, collinearity, etc.
#             if is_merge_candidate(line, lines[j], angle_thresh, dist_thresh=search_dist, collinearity_thresh=collinearity_thresh):
#                 # Build bridge + merged line
#                 merged_line, connector = connect_lines_by_bridge([line, lines[j]])
#
#                 # Replace line i with merged
#                 lines[i] = merged_line
#
#                 # Remove line j
#                 lines.pop(j)
#                 match_found = True
#                 break  # restart with updated line[i]
#
#         if not match_found:
#             i += 1  # move to next line
#
#     return lines

# def flatten_multilines(gdf):
#     """
#     Converts MultiLineStrings into separate LineStrings.
#     """
#     from shapely.geometry import MultiLineString
#     single_parts = []
#     for geom in gdf.geometry:
#         if geom.geom_type == 'LineString':
#             single_parts.append(geom)
#         elif geom.geom_type == 'MultiLineString':
#             single_parts.extend(list(geom.geoms))
#         else:
#             print(f"Unsupported geometry type: {geom.geom_type}")
#     print('gdf flattened')
#     return gpd.GeoDataFrame(geometry=single_parts, crs=gdf.crs)

# def locally_merge_lines(lines, search_dist=10.0, angle_thresh=3, collinearity_thresh=0.5):
#     lines = list(lines)
#     connectors = []
#     i = 0
#     while i < len(lines):
#         line = lines[i]
#         match_found = False
#         for j in range(len(lines)):
#             if i == j:
#                 continue
#             if line.distance(lines[j]) > search_dist:
#                 continue
#             if is_merge_candidate(line, lines[j], angle_thresh, search_dist, collinearity_thresh):
#                 merged_line, connector = connect_lines_by_bridge([line, lines[j]])
#                 lines[i] = merged_line
#                 lines.pop(j)
#                 connectors.append(connector)
#                 match_found = True
#                 break
#         if not match_found:
#             i += 1
#     return lines, connectors

# def find_merge_pairs(gdf, angle_thresh=16, dist_thresh=2.0, collinearity_thresh=0.5):
#     merge_pairs = []
#     for i, line1 in enumerate(gdf.geometry):
#         for j, line2 in enumerate(gdf.geometry):
#             if i >= j:
#                 continue
#             if is_merge_candidate(line1, line2, angle_thresh, dist_thresh, collinearity_thresh):
#                 merge_pairs.append((i, j))
#     return merge_pairs