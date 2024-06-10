# This file is part of hvsrprocpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

# Copyright (c) 2024 Francisco Javier Ornelas (jornela1@g.ucla.edu)

#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.

"""Functions to help in hvsr processing."""

import os
import obspy
import numpy as np
import pandas as pd

__all__ = ['add_mseed_tool', 'proc_mseed_data', 'proc_txt_data', 'process_time_series']


def add_mseed_tool(st):
    """

    Function to trim traces if necessary.

    Parameters
    ----------

    st: object
        Stream object containing traces to trim.

    returns trimmed stream.

    """
    while True:
        # Plot the current stream
        st.plot()

        in_start_time = st[0].stats.starttime
        in_end_time = st[0].stats.endtime

        print("Current start time:", in_start_time)
        print("Current end time:", in_end_time)

        new_start_time = input("Enter the new start time (YYYY-MM-DDTHH:MM:SS): ")
        new_end_time = input("Enter the new end time (YYYY-MM-DDTHH:MM:SS): ")

        # Convert input times to UTCDateTime objects
        start_time = obspy.UTCDateTime(new_start_time)
        end_time = obspy.UTCDateTime(new_end_time)

        # Trim the stream
        st.trim(starttime=start_time, endtime=end_time)

        # Replot the trimmed stream
        st.plot()

        continue_trim = input("Do you want to continue trimming? (y/n): ")
        if continue_trim.lower() != 'y':
            break

    return st


def proc_mseed_data(file_direc, h1_fn, h2_fn, v_fn, trim_flag=False, time_cut=300):
    """

    :param file_direc:
    :param h1_fn:
    :param h2_fn:
    :param v_fn:
    :param trim_flag:
    :param time_cut:
    :return:
    """
    h1 = obspy.read(os.path.join(file_direc, h1_fn))
    h2 = obspy.read(os.path.join(file_direc, h2_fn))
    v = obspy.read(os.path.join(file_direc, v_fn))

    h1.merge(method=1, interpolation_samples=-1)
    h2.merge(method=1, interpolation_samples=-1)
    v.merge(method=1, interpolation_samples=-1)

    if trim_flag:
        h1 = add_mseed_tool(h1)
        h2 = add_mseed_tool(h2)
        v = add_mseed_tool(v)

    dt = float(h1[0].stats.delta)

    h1 = h1[0].data
    h2 = h2[0].data
    v = v[0].data

    npts_min = min(len(h1), len(h2), len(v))

    time = np.dot(range(npts_min), dt)

    h1 = h1[:npts_min]
    h2 = h2[:npts_min]
    v = v[:npts_min]

    tcut_val = time_cut

    npts_cut = int(tcut_val / dt)

    h1 = h1[(npts_cut + 1):(npts_min - npts_cut)]
    h2 = h2[(npts_cut + 1):(npts_min - npts_cut)]
    v = v[(npts_cut + 1):(npts_min - npts_cut)]

    return h1, h2, v, dt, time


def proc_txt_data(file_direc, h1_fn, h2_fn, v_fn, time_cut=300):
    """

    :param file_direc:
    :param h1_fn:
    :param h2_fn:
    :param v_fn:
    :param time_cut:
    :return:
    """
    h1 = pd.read_csv(os.path.join(file_direc, h1_fn), sep=' ', header=None, names=['time', 'vel'])
    h2 = pd.read_csv(os.path.join(file_direc, h2_fn), sep=' ', header=None, names=['time', 'vel'])
    v = pd.read_csv(os.path.join(file_direc, v_fn), sep=' ', header=None, names=['time', 'vel'])

    # truncate records to the smallest length
    npts_min = min(len(v), len(h1), len(h2))

    # Extract columns from dataframe
    time = h1['time'].values
    h1 = h1['vel'].values
    h2 = h2['vel'].values
    v = v['vel'].values

    time = time[:npts_min]
    h1 = h1[:npts_min]
    h2 = h2[:npts_min]
    v = v[:npts_min]

    # Specify time step
    dt = time[1] - time[0]

    tcut_val = time_cut

    npts_cut = int(tcut_val / dt)

    h1 = h1[(npts_cut + 1):npts_min - npts_cut]
    h2 = h2[(npts_cut + 1):npts_min - npts_cut]
    v = v[(npts_cut + 1):npts_min - npts_cut]

    return h1, h2, v, dt, time


def process_time_series(h1_fn, h2_fn, v_fn, directory, **kwargs):
    """

    Parameters
    ----------
    h1_fn: string
        filename for h1.
    h2_fn: string
        filename for h2.
    v_fn: string
        filename for v.
    directory: string
        directory where components are stored.

    **kwargs
        time_cut : int
            Integer representing how much time to be cut from the beginning
            and end of the amplitude component of a  time series array.
        file_type :  int
            Indicates whether file is of mseed or txt file type. default = 1 (mseed)

    returns arrays of time series components

    """

    time_cut = kwargs.get('time_cut', 300)
    file_type = kwargs.get('file_type', 1)
    trim_flag = kwargs.get('trim_flag', False)

    if file_type == 1:
        h1, h2, v, dt, time = proc_mseed_data(file_direc=directory, h1_fn=h1_fn,
                                              h2_fn=h2_fn, v_fn=v_fn, trim_flag=trim_flag,
                                              time_cut=time_cut)

    elif file_type == 2:
        h1, h2, v, dt, time = proc_txt_data(file_direc=directory, h1_fn=h1_fn,
                                            h2_fn=h2_fn, v_fn=v_fn, time_cut=time_cut)
    else:
        raise ValueError("Invalid file type used. Use either 1 for mseed or 2 for text file")

    return h1, h2, v, dt, time
