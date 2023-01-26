import pyresample 

def resample(target_grid, original_grid, variable):
    '''
    Resamples (i.e., re-projects, re-grid) the original grid to the target grid 
    using a nearest neighborhood approach
    Args:
        target_grid, 2-tuple of 2D arrays of target latitude and longitude 
        original_grid, 2-tuple of 2D arrays of original latitude and longitude 
        variable, 2D array to be resampled
    Return:
        variable_nearest, 2D array of variable resampled to the target grid
    '''
    # Create a pyresample object holding the original grid
    orig_def = pyresample.geometry.SwathDefinition(lons=original_grid[1], lats=original_grid[0])

    # Create another pyresample object for the target grid
    targ_def = pyresample.geometry.SwathDefinition(lons=target_grid[1], lats=target_grid[0])

    variable_nearest = pyresample.kd_tree.resample_nearest(orig_def, variable, \
                    targ_def, radius_of_influence=50000, fill_value=None)

    return variable_nearest