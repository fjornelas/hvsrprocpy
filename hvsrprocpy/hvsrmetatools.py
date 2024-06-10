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

"""Class definition for HvsrMetaTools object."""

import os
import pandas as pd
from obspy import read_inventory
import numpy as np

__all__ = ['HvsrMetaTools']


class HvsrMetaTools:

    """

    Set of functions that processes metadata from output of hvsrprocpy

    """

    @staticmethod
    def process_xml_file(xml_file_path):
        """
        Function that parses through .xml file data.

        Parameters
        ----------

        xml_file_path: string
            Directory where xml file is stored.

        returns the station and channel information

        """

        # Read the inventory from the xml file
        inv = read_inventory(xml_file_path, format="STATIONXML")

        # Search through network information
        net = inv[0]

        # Search through station information
        sta = net[0]

        # Search through channel information
        cha = sta[0]

        return sta, cha

    @staticmethod
    def add_to_dataframe(df, sta, cha, field_crew, user, weather, ground_type, sensor_ground_coupling,
                         monochromatic, building, transients, data_type, sensor_corner_frequency,
                         gain, comments):
        """
        Function that adds xml information to dataframe.

        Parameters
        ----------

        df: list
            Empty dataframe list
        sta: dictionary
            Station information
        cha: dictionary
            Channel information
        field_crew: string
            Indicates who was at the site. Typ. Name
        user: string
            Indicates who is generating the dataframe.
        weather: string
            Indicates the type of weather. Ex. Sunny
        ground_type: string
            Indicates where the sensor is place. Ex. Concrete.
        sensor_ground_coupling: string
            Indicates whether the sensor was buried or placed on the ground. Ex. Buried
        monochromatic: string
            Indicates whether there was monochromatic noise present.
        building: string
            Indicates whether there was a building
        transients: string
            Indicates whether there were transients present during the test.
        data_type: int
            Indicates whether the data is microtremor or earthquake data.
            0 is microtremor, 1 is earthquake.
        sensor_corner_frequency: float
            Indicates the sensors corner frequency.
        gain: int
            Indicates the gain of the sensor used.
        comments: string
            Indicates other comments.

        returns dataframe of the metadata.

        """
        if cha.data_logger:
            recorder = f"{cha.data_logger.description} {cha.data_logger.model}"
        else:
            recorder = None

        sensor_type = None

        if cha.sensor:
            if cha.sensor.description:
                sensor_type = cha.sensor.description
            else:
                sensor_type = cha.sensor.type
            sensor_manufacturer = cha.sensor.manufacturer
            sensor_serial_number = cha.sensor.serial_number
        else:
            sensor_manufacturer = None
            sensor_serial_number = None

        if sta.site:
            station_name = sta.description
        else:
            station_name = sta.site.name

        azimuth = 0

        df.loc[len(df)] = [
            station_name, cha.longitude, cha.latitude, cha.elevation,
            str(cha.start_date).split()[0][:10], str(cha.end_date).split()[0][:10],
            str(cha.start_date).split()[0][11:25], str(cha.end_date).split()[0][11:25],
            field_crew, recorder,
            cha.data_logger.serial_number if cha.data_logger else None,
            gain, sensor_manufacturer,
            sensor_type, sta.code, sensor_serial_number,
            sensor_corner_frequency, weather, ground_type, monochromatic,
            sensor_ground_coupling, building, transients, azimuth,
            user, comments, data_type, None, None, None, None
        ]
        return df

    @staticmethod
    def process_hvsr_metadata(field_crew, user, weather, ground_type, sensor_ground_coupling,
                              monochromatic, building, transients, data_type, sensor_corner_frequency,
                              gain, comments, directory):
        """
        Function that combines xml and dataframe function to generate a
        metadata table for multiple stations.

        Parameters
        ----------

        field_crew: string
            Indicates who was at the site. Typ. Name
        user: string
            Indicates who is generating the dataframe.
        weather: string
            Indicates the type of weather. Ex. Sunny
        ground_type: string
            Indicates where the sensor is place. Ex. Concrete.
        sensor_ground_coupling: string
            Indicates whether the sensor was buried or placed on the ground. Ex. Buried
        monochromatic: string
            Indicates whether there was monochromatic noise present.
        building: string
            Indicates whether there was a building
        transients: string
            Indicates whether there were transients present during the test.
        data_type: int
            Indicates whether the data is microtremor or earthquake data.
            0 is microtremor, 1 is earthquake.
        sensor_corner_frequency: float
            Indicates the sensors corner frequency.
        gain: int
            Indicates the gain of the sensor used.
        comments: string
            Indicates other comments.
        directory: string
            Directory where .xml files are stored.

        returns metadata dataframe.

        """

        df_columns = ['name', 'longitude', 'latitude', 'elevation', 'start_date', 'end_date', 'start_time',
                      'end_time', 'field_crew', 'seismic_recorder', 'recorder_serial_number', 'gain',
                      'sensor_manufacturer', 'sensor', 'sensor_name', 'sensor_serial_number', 'sensor_corner_frequency',
                      'weather', 'ground_type', 'monochromatic_noise_source', 'sensor_ground_coupling', 'building',
                      'transients', 'azimuth', 'user', 'comments', 'data_type', 'mass_position_w', 'mass_position_v',
                      'mass_position_u', 'mag_dec']

        df = pd.DataFrame(columns=df_columns)

        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.endswith('.xml'):
                    try:
                        sta, cha = HvsrMetaTools.process_xml_file(os.path.join(root, name))
                        df = HvsrMetaTools.add_to_dataframe(df, sta, cha, field_crew, user, weather,
                                                            ground_type, sensor_ground_coupling,
                                                            monochromatic, building, transients,
                                                            data_type, sensor_corner_frequency,
                                                            gain, comments)
                    except Exception as e:
                        print(f"Error processing {name}: {e}")

        return df

    @staticmethod
    def process_hvsr_metadata_single_site(field_crew, user, weather, ground_type, sensor_ground_coupling,
                                          monochromatic=None, building=None, transients=None, data_type=0,
                                          sensor_corner_frequency=0.0083, gain=1, comments=None, azimuth=0,
                                          inventory_path=None):
        """

       Function that combines xml and dataframe function to generate a
        metadata table for a single station.

        Parameters
        ----------

        field_crew: string
            Indicates who was at the site. Typ. Name
        user: string
            Indicates who is generating the dataframe.
        weather: string
            Indicates the type of weather. Ex. Sunny
        ground_type: string
            Indicates where the sensor is place. Ex. Concrete.
        sensor_ground_coupling: string
            Indicates whether the sensor was buried or placed on the ground. Ex. Buried
        monochromatic: string
            Indicates whether there was monochromatic noise present.
        building: string
            Indicates whether there was a building
        transients: string
            Indicates whether there were transients present during the test.
        data_type: int
            Indicates whether the data is microtremor or earthquake data.
            0 is microtremor, 1 is earthquake.
        sensor_corner_frequency: float
            Indicates the sensors corner frequency.
        gain: int
            Indicates the gain of the sensor used.
        comments: string
            Indicates other comments.
        azimuth: int
            Degree of north azimuth. Default = 0
        inventory_path: string
            Indicates directory where xml files are stored.

        returns metadata dataframe.

        """

        df = pd.DataFrame(
            columns=['site_name', 'longitude', 'latitude', 'elevation', 'start_date', 'end_date', 'start_time',
                     'end_time', 'field_crew', 'seismic_recorder', 'recorder_serial_number', 'gain',
                     'sensor_manufacturer', 'sensor', 'sensor_name', 'sensor_serial_number',
                     'sensor_corner_frequency', 'weather', 'ground_type', 'monochromatic_noise_source',
                     'sensor_ground_coupling', 'building', 'transients', 'azimuth', 'user', 'comments',
                     'data_type', 'mass_position_w', 'mass_position_v', 'mass_position_u', 'mag_dec'])

        if inventory_path:
            inv = read_inventory(inventory_path, format="STATIONXML")
            net = inv[0]
            sta = net[0]
            cha = sta[0]

            if cha.data_logger:
                recorder = f"{cha.data_logger.description} {cha.data_logger.model}"
            else:
                recorder = None

            sensor_type = None

            if cha.sensor:
                if cha.sensor.description:
                    sensor_type = cha.sensor.description
                else:
                    sensor_type = cha.sensor.type
                sensor_manufacturer = cha.sensor.manufacturer
                sensor_serial_number = cha.sensor.serial_number
            else:
                sensor_manufacturer = None
                sensor_serial_number = None

            if sta.site:
                station_name = sta.description
            else:
                station_name = sta.site.name

            df.loc[len(df)] = [
                station_name, cha.longitude, cha.latitude, cha.elevation,
                str(cha.start_date).split()[0][:10], str(cha.end_date).split()[0][:10],
                str(cha.start_date).split()[0][11:25], str(cha.end_date).split()[0][11:25],
                field_crew, recorder,
                cha.data_logger.serial_number if cha.data_logger else "Unknown",
                gain, sensor_manufacturer,
                sensor_type, sta.code, sensor_serial_number,
                sensor_corner_frequency, weather, ground_type, monochromatic,
                sensor_ground_coupling, building, transients, azimuth,
                user, comments, data_type, None, None, None, None
            ]

        else:
            pass

        return df

    @staticmethod
    def create_mean_curves_csv(folder_path, filename, add_sim_path, output_flag=False):

        """
        Function to combine multiple .csv files of mean curves from multiple stations.

        folder_path: string
            Specifies where all the station files are stored.
        filename: string
            Indicates the filename for the new .csv file to be called
        add_sim_path: string
            Indicates a unique directory structure that other folders in the folder path contain.

        returns a dataframe of combined mean curve csv files.

        """

        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

        max_len = 0  # Variable to store the maximum length of mean curve data

        for folder in folders:
            folder_full_path = os.path.join(folder_path, folder)
            mean_curve_file = os.path.join(folder_full_path, add_sim_path, 'Test_hvsr_mean.csv')

            if not os.path.isfile(mean_curve_file):  # Check if the mean curve file exists
                continue

            mean_curve_df = pd.read_csv(mean_curve_file)

            max_len = max(max_len, len(mean_curve_df))

        hvsr_mean_all_df = np.full((max_len, len(folders) * 2 + 1), np.nan)  # Initialize array for mean and sd data

        for folder_idx, folder in enumerate(folders):
            folder_full_path = os.path.join(folder_path, folder)
            mean_curve_file = os.path.join(folder_full_path, add_sim_path, 'Test_hvsr_mean.csv')

            if not os.path.isfile(mean_curve_file):  # Check if the mean curve file exists
                continue

            mean_curve_df = pd.read_csv(mean_curve_file)

            hvsr_mean_all_df[:len(mean_curve_df), 0] = mean_curve_df['freq_Hz']  # Assign frequency data

            for i, (mean_data, sd_data) in enumerate(zip(mean_curve_df['HVSR mean'], mean_curve_df['HVSR sd'])):
                hvsr_mean_all_df[i, 1 + folder_idx * 2] = mean_data
                hvsr_mean_all_df[i, 1 + folder_idx * 2 + 1] = sd_data

        colnames = ['freq_Hz']  # Adding 'frequency' as the first column name
        for folder_name in folders:
            colnames.extend([f'HVSR_mean_{folder_name}', f'HVSR_sd_{folder_name}'])

        df = pd.DataFrame(hvsr_mean_all_df, columns=colnames)

        if output_flag:
            output_csv = os.path.join(folder_path, filename)
            df.to_csv(output_csv, index=False, header=True)
        else:
            pass

        return df

    @staticmethod
    def combine_metadata(folder_path, add_sim_path):
        """

        Function to combine multiple processing metadata files.

        Parameters:
        ----------

        folder_path: string
            Indicates the directory where all station folders are stored.
        add_sim_path: string
            Indicates a unique folder structure that is contained in all station folders.

        returns a dataframe of combined metadata csv files.

        """
        df_list = []  # List to store DataFrames for each folder

        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

        for folder in folders:
            station_name = folder  # Folder name will be used as station name
            folder_full_path = os.path.join(folder_path, folder)
            meta_file = os.path.join(folder_full_path, add_sim_path, 'Test_metadata.csv')

            if not os.path.isfile(meta_file):  # Check if the metadata file exists
                continue

            meta_df = pd.read_csv(meta_file)
            # Add station name column
            meta_df.insert(0, 'name', station_name)
            df_list.append(meta_df)

        # Combine all DataFrames into one
        combined_df = pd.concat(df_list, ignore_index=True)

        return combined_df
