# _HvsrMetaTools_ (HVSR Metadata Tools)

Class definition for HvsrMetaTools, a class containing functions
that can be used to process metadata files such as station.xml files,
which are output from 3-component seismometers.

`class HvsrMetaTools`

`Bases: object`

Class containing multiple static functions that 
process station.xml or other metadata files.

The definitions for each function are below:

## _process_xml_file_

Function that parses through .xml file data.

`process_xml_file(xml_file_path)`

`Parameters:`

**xml_file_path** (string) - Directory where xml file is stored.

`returns:` 

The station and channel information

## _add_to_dataframe_

Function that adds xml information to dataframe.

add_to_dataframe(df, sta, cha, field_crew, user, weather, ground_type, sensor_ground_coupling,
                         monochromatic, building, transients, data_type, sensor_corner_frequency,
                         gain, comments)

`Parameters:`

**df** (list) - Empty dataframe list

**sta** (dictionary) - Station information

**cha** (dictionary) - Channel information

**field_crew** (string) - Indicates who was at the site. Typ. Name

**user** (string) - Indicates who is generating the dataframe.

**weather** (string) - Indicates the type of weather. Ex. Sunny

**ground_type** (string) - Indicates where the sensor is place. Ex. Concrete.

**sensor_ground_coupling** (string) - Indicates whether the sensor was buried or placed on the ground. Ex. Buried

**monochromatic** (string) - Indicates whether there was monochromatic noise present.

**building** (string) - Indicates whether there was a building

**transients** (string) - Indicates whether there were transients present during the test.

**data_type** (int) - Indicates whether the data is microtremor or earthquake data. 0 is microtremor, 1 is earthquake.

**sensor_corner_frequency** (float) - Indicates the sensors corner frequency.

**gain** (int) - Indicates the gain of the sensor used.

**comments** (string) - Indicates other comments.

`returns:` 

Dataframe of the metadata.


## _process_hvsr_metadata_

Function that combines xml and dataframe function to generate a
        metadata table for multiple stations.

`process_hvsr_metadata(field_crew, user, weather, ground_type, sensor_ground_coupling, monochromatic, building, transients, data_type, sensor_corner_frequency, gain, comments, directory)`

`Parameters:`

**field_crew** (string) - Indicates who was at the site. Typ. Name

**user** (string) - Indicates who is generating the dataframe.

**weather** (string) - Indicates the type of weather. Ex. Sunny

**ground_type** (string) - Indicates where the sensor is place. Ex. Concrete.

**sensor_ground_coupling** (string) - Indicates whether the sensor was buried or placed on the ground. Ex. Buried

**monochromatic** (string) - Indicates whether there was monochromatic noise present.

**building** (string) - Indicates whether there was a building

**transients** (string) - Indicates whether there were transients present during the test.

**data_type** (int) - Indicates whether the data is microtremor or earthquake data. 0 is microtremor, 1 is earthquake.

**sensor_corner_frequency** (float) - Indicates the sensors corner frequency.

**gain** (int) - Indicates the gain of the sensor used.

**comments** (string) - Indicates other comments.

**directory** (string) - Directory where .xml files are stored.

`returns:` 

Metadata dataframe.

## _process_hvsr_metadata_single_site_

Function that combines xml and dataframe function to generate a
metadata table for a single station.

`process_hvsr_metadata_single_site(field_crew, user, weather, ground_type, sensor_ground_coupling, monochromatic=None, building=None, transients=None, data_type=0, sensor_corner_frequency=0.0083, gain=1, comments=None, azimuth=0, inventory_path=None)`

`Parameters`

**field_crew** (string) - Indicates who was at the site. Typ. Name

**user** (string) - Indicates who is generating the dataframe.

**weather** (string) - Indicates the type of weather. Ex. Sunny

**ground_type** (string) - Indicates where the sensor is place. Ex. Concrete.

**sensor_ground_coupling** (string) - Indicates whether the sensor was buried or placed on the ground. Ex. Buried

**monochromatic** (string) - Indicates whether there was monochromatic noise present.

**building** (string) - Indicates whether there was a building

**transients** (string) - Indicates whether there were transients present during the test.

**data_type** (int) - Indicates whether the data is microtremor or earthquake data. 0 is microtremor, 1 is earthquake.

**sensor_corner_frequency** (float) - Indicates the sensors corner frequency.

**gain** (int) - Indicates the gain of the sensor used.

**comments** (string) - Indicates other comments.

**azimuth** (int) - Degree of north azimuth. Default = 0

**inventory_path** (string) - Indicates directory where xml files are stored.

`returns:` 

Metadata dataframe.

## create_mean_curves_csv

Function to combine multiple .csv files of mean curves from multiple stations.

`create_mean_curves_csv(folder_path, filename, add_sim_path, output_flag=False)`

`Parameters:`

**folder_path** (string) - Specifies where all the station files are stored.

**filename** (string) - Indicates the filename for the new .csv file to be called

**add_sim_path** (string) - Indicates a unique directory structure that other folders in the folder path contain.

`returns:` 

A dataframe of combined mean curve csv files.


## combine_metadata

Function to combine multiple processing metadata files.

`combine_metadata(folder_path, add_sim_path)`

`Parameters:`

**folder_path** (string) - Indicates the directory where all station folders are stored.

**add_sim_path** (string) - Indicates a unique folder structure that is contained in all station folders.

`returns:` 

A dataframe of combined metadata csv files.

