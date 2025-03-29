from LineConnect import *

lines = [
    LineString([(0, 0), (5, 5)]),   # Line 0
    LineString([(5.2, 5.2), (10, 10)]),  # Line 1- near 0, same direction
    LineString([(0, 5), (5, 0)]),   # Line 2 - crosses 0, different direction
    LineString([(11, 10), (15, 14)]), # Line 3 - far, but same angle
    LineString([(6, 7), (6, 11)]),  # Line 4 - parallel/near 1 but vertical
    LineString([(5, 7), (5, 11)]),  # Line 5 - vertical, near 1
    LineString([(5, 11.5),(6, 15)]),  # Line 6 -
    LineString([(10.2, 10.2),(12.2,12.2 )]),  # Line 7
    LineString([(12.5, 12.5), (15.4, 15.2)]), # Line 8
    LineString([(6, 12.3), (7, 15.4)]), # Line 9
    LineString([(6, 0), (12, 1)]), # Line 10
    ]

gdf = gpd.GeoDataFrame(geometry=lines)

def run_LinCon_demo(search_dist=5, angle_thresh=20,collinearity_thresh=1, gdf = gdf ):
    # Create synthetic lines for testing
    plot_lines(gdf, annotate=True)

    new_lines, connectors = locally_merge_lines(gdf, search_dist, angle_thresh, collinearity_thresh)
    gdf_new = gpd.GeoDataFrame(geometry=new_lines, crs=gdf.crs)
    connectors_gdf = gpd.GeoDataFrame(geometry=connectors, crs=gdf.crs)

    # Save connectors to file
    # connectors_gdf.to_file("connectors_only.shp")
    # gdf_new.to_file("Merged_lines.shp")

    logging.basicConfig(filename='merge_log.txt', level=logging.INFO)

    logging.info("Started processing")
    logging.info(f"Input CRS: {gdf.crs}")
    logging.info(f"Number of features: {len(gdf)}")

    # Plot
    plot_results(gdf, gdf_new, connectors_gdf, search_dist, angle_thresh, collinearity_thresh, annotate=True)

    # Metadata Report
    report_metadata(gdf, gdf_new, search_dist, angle_thresh, collinearity_thresh)




def demo_func_angle_between_lines(lines):
    gdf = gpd.GeoDataFrame(geometry=lines)
    plot_lines(gdf)
    angles_list = []  # initiate an empty list to store the angles
    for i, line1 in enumerate(lines):  # Loop through each line
        for j, line2 in enumerate(lines):  # Compare with every other line
            # Skip the same line or if the lines are not close enough
            if i >= j:
                continue
            angle = angle_between_lines(line1, line2)
            entry = f"Angle between line {i} and line {j}: {angle:.2f} degrees"
            angles_dict = {
                'line1': i,
                'line2': j,
                'angle': angle,
                'entry': entry
            }
            angles_list.append(angles_dict)

    print(angles_list[1:10])  # Print the first 10 angles for brevity


# # do some checks to ensure the collinearity score is working as expected
# score = directional_collinearity_score(lines[0], lines[1]) # should be 0
# print(f"Directional collinearity score between line {0} and line {1}: {score:.2f}")
# score = directional_collinearity_score(lines[1], lines[3]) # should be 0.765
# print(f"Directional collinearity score between line {1} and line {3}: {score:.2f}")
# score = directional_collinearity_score(lines[4], lines[6]) # should be 0
# print(f"Directional collinearity score between line {4} and line {6}: {score:.2f}")
# score = directional_collinearity_score(lines[5], lines[6]) # should be 0
# print(f"Directional collinearity score between line {5} and line {6}: {score:.2f}")
#
# is_merge_candidate(lines[0], lines[1],angle_thresh=0.5, dist_thresh=0.3, collinearity_thresh=0.1) # should be true even at very low values
# is_merge_candidate(lines[1], lines[3],angle_thresh=10, dist_thresh=2.0, collinearity_thresh=1.0)  # should be true
# is_merge_candidate(lines[1], lines[3],angle_thresh=0, dist_thresh=2.0, collinearity_thresh=0.5) # should be false as we decrease colinearity threshold
# is_merge_candidate(lines[1], lines[3],angle_thresh=0, dist_thresh=1.0, collinearity_thresh=1) # should be false as we decrease distance threshold
# is_merge_candidate(lines[4], lines[9],angle_thresh=20, dist_thresh=3, collinearity_thresh=1) # should be false as we decrease distance threshold

print("LineConnectChecks.py has been imported, try to run: run_LinCon_demo()")

#run_LinCon_demo(angle_thresh=16)