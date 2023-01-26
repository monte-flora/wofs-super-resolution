import numpy as np
from scipy.fft import fft2, ifft2

class SpatialFilter:
    """SpatialFilter uses scipy.fft and a Butterworth filter to remove spatial scales above
    and below a given maximum and minimum resolution, respectively.
    
    Code is from thunderhoser (URL).
    
    Attributes
    ---------------
    grid_spacing : int 
        Grid spacing (in meters)
        
    min_resolution : int 
        Minimumal scales to preserve (in meters). 
        
    max_resolution : int (default=np.inf)
        Maximum scales to preserve (in meters).
        Default is infinity so that all large scales are preserved.
    
    filter_order : int (default=10)
        Filter order for the Butterworth filter.
    
    """
    TOLERANCE = 1e-6
    
    def __init__(self, grid_spacing, min_resolution, max_resolution=np.inf, filter_order=4):
        self.grid_spacing_ = grid_spacing
        self.min_resolution_ = min_resolution
        self.max_resolution_ = max_resolution 
        self.filter_order_ = filter_order

    def taper_spatial_data(self, spatial_data_matrix):
        """Tapers spatial data by putting zeros along the edge.
        M = number of rows in grid
        N = number of columns in grid
        :param spatial_data_matrix: M-by-N np array of real numbers.
        :return: spatial_data_matrix: Same but after tapering.
        """
        num_rows = spatial_data_matrix.shape[0]
        num_columns = spatial_data_matrix.shape[1]
    
        # If the number is even, make it odd
        row_pad = 1 if num_rows%2 == 0 else 0
        col_pad = 1 if num_columns%2 == 0 else 0

        padding_arg = (
            (row_pad, 0),
            (col_pad, 0)
        )

        spatial_data_matrix = np.pad(
            spatial_data_matrix, pad_width=padding_arg, mode='constant',
            constant_values=0.
        )

        return spatial_data_matrix


    def _get_spatial_resolutions(self, num_grid_rows, num_grid_columns):
        """Computes spatial resolution for each Fourier coefficient.
        M = number of rows in spatial grid
        N = number of columns in spatial grid
        Matrices returned by this method correspond to matrices of Fourier
        coefficients returned by `np.fft.fft2`.  The x-coordinate increases with
        column index, and the y-coordinate increases with row index.
        :param num_grid_rows: M in the above discussion.
        :param num_grid_columns: N in the above discussion.
        :param grid_spacing_metres: Grid spacing (for which I use "resolution" as a
            synonym).
        :return: x_resolution_matrix_metres: M-by-N np array of resolutions in
            x-direction.
        :return: y_resolution_matrix_metres: Same but for y-direction.
        """
        num_half_rows_float = float(num_grid_rows - 1) / 2
        num_half_rows = int(np.round(num_half_rows_float))

        num_half_columns_float = float(num_grid_columns - 1) / 2
        num_half_columns = int(np.round(num_half_columns_float))

        # Find resolutions in x-direction.
        unique_x_wavenumbers = np.linspace(
            0, num_half_columns, num=num_half_columns + 1, dtype=int
        )
        x_wavenumbers = np.concatenate((
            unique_x_wavenumbers, unique_x_wavenumbers[1:][::-1]
        ))
        x_wavenumber_matrix = np.expand_dims(x_wavenumbers, axis=0)
        x_wavenumber_matrix = np.repeat(
            x_wavenumber_matrix, axis=0, repeats=num_grid_rows
        )

        with np.errstate(divide='ignore'):
            x_grid_length_metres = self.grid_spacing_ * (num_grid_columns - 1)
            x_resolution_matrix_metres = (
            0.5 * x_grid_length_metres / x_wavenumber_matrix
            )

        # Find resolutions in y-direction.
        unique_y_wavenumbers = np.linspace(
            0, num_half_rows, num=num_half_rows + 1, dtype=int
        )
        y_wavenumbers = np.concatenate((
            unique_y_wavenumbers, unique_y_wavenumbers[1:][::-1]
        ))
        y_wavenumber_matrix = np.expand_dims(y_wavenumbers, axis=1)
        y_wavenumber_matrix = np.repeat(
            y_wavenumber_matrix, axis=1, repeats=num_grid_columns
        )
    
        with np.errstate(divide='ignore'):
            y_grid_length_metres = self.grid_spacing_ * (num_grid_rows - 1)
            y_resolution_matrix_metres = (
                0.5 * y_grid_length_metres / y_wavenumber_matrix
            )

        return x_resolution_matrix_metres, y_resolution_matrix_metres


    def apply_rectangular_filter(self, coefficient_matrix):
        """Applies rectangular band-pass filter to Fourier coefficients.
        M = number of rows in spatial grid
        N = number of columns in spatial grid
        :param coefficient_matrix: M-by-N np array of coefficients in format
            returned by `np.fft.fft2`.

        :return: coefficient_matrix: Same as input but maybe with some coefficients
            zeroed out.
        """

        # Do actual stuff.
        x_resolution_matrix_metres, y_resolution_matrix_metres = (
            self._get_spatial_resolutions(
            num_grid_rows=coefficient_matrix.shape[0],
            num_grid_columns=coefficient_matrix.shape[1],
            )
        )

        resolution_matrix_metres = np.sqrt(
            x_resolution_matrix_metres ** 2 + y_resolution_matrix_metres ** 2
        )

        coefficient_matrix[resolution_matrix_metres > self.max_resolution_] = 0.
        coefficient_matrix[resolution_matrix_metres < self.min_resolution_] = 0.
        
        return coefficient_matrix


    def apply_butterworth_filter(self, coefficient_matrix, ):
        """Applies Butterworth band-pass filter to Fourier coefficients.
        :param coefficient_matrix: See doc for `apply_rectangular_filter`.
        :return: coefficient_matrix: Same as input but after filtering.
        """

        # Determine horizontal, vertical, and total wavenumber for each Fourier
        # coefficient.
        x_resolution_matrix_metres, y_resolution_matrix_metres = (
            self._get_spatial_resolutions(
                num_grid_rows=coefficient_matrix.shape[0],
                num_grid_columns=coefficient_matrix.shape[1],
            )
        )

        x_wavenumber_matrix_metres01 = (2 * x_resolution_matrix_metres) ** -1
        y_wavenumber_matrix_metres01 = (2 * y_resolution_matrix_metres) ** -1
        wavenumber_matrix_metres01 = np.sqrt(
            x_wavenumber_matrix_metres01 ** 2 + y_wavenumber_matrix_metres01 ** 2
        )

        # High-pass part.
        if not np.isinf(self.max_resolution_):
            min_wavenumber_metres01 = (2 * self.max_resolution_metres_) ** -1
            ratio_matrix = wavenumber_matrix_metres01 / min_wavenumber_metres01
            gain_matrix = 1 - (1 + ratio_matrix ** (2 * self.filter_order_)) ** -1
            coefficient_matrix = coefficient_matrix * gain_matrix

        # Low-pass part.
        if self.min_resolution_ > self.grid_spacing_:
            max_wavenumber_metres01 = (2 * self.min_resolution_) ** -1
            ratio_matrix = wavenumber_matrix_metres01 / max_wavenumber_metres01
            gain_matrix = (1 + ratio_matrix ** (2 * self.filter_order_)) ** -1
            coefficient_matrix = coefficient_matrix * gain_matrix

        return coefficient_matrix


    def filter(self, data):
        """Applies a fourier transform, filters out smaller scales using a butterworth filter, 
        and then converts back into the original space
    
        Parameters
        --------------
        data : array-like of shape (NY, NX)
            Input data.
    
        grid_spacing : int
            Grid spacing (in meters)
        
        min_resolution : int 
            Minimal grid spacing to preserve. Scales below this grid spacing
            are filtered out. 
    
        Returns
        -----------
        array-like of shape (NY, NX)
            Data with scales below the minimal resolution filtered out. 
        """
        # Pad the data for the prior to filtering.
        data_pad = self.taper_spatial_data(data)

        # Fourier transform the data.
        coefficient_matrix = fft2(data_pad)

        # Apply the filtering in the transformed space.
        new_coef_mat = self.apply_butterworth_filter(coefficient_matrix)

        # Convert back to the original space (np.real converts from complex128 to float)
        return np.real(ifft2(new_coef_mat))


