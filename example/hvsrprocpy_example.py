#!/usr/bin/env python
# coding: utf-8

# <h1><center> 
#      Example of hvsrprocpy Usage
# </center></h1>
# <center>
#     By: Francisco Javier G. Ornelas (jornela1@ucla.edu)<br>Originally created: Apr. 28, 2024<br>
# </center>
# <center>
#     This jupyter notebook will walk you through the various functions that are contained in this library.
# </center>

# # Download the python library if not already done so

# In[2]:


# pip install hvsrprocpy


# # Load in the necessary python libraries

# In[2]:


#Import the hvsrprocpy package
import os
import hvsrprocpy as hv
import pymysql, json
import pandas as pd
from hvsrprocpy import HvsrMetaTools as hmt


# # If you need help, see below:

# In[5]:


#Print statement to see what are the functions and inputs within hvsrProc
# help(hv.hvsr)


# # Process the microtremor or earthquake data and get HVSR

# In[21]:


get_ipython().run_cell_magic('time', '', "\n#Specify directory where .txt or .mseed files are stored\ndirec = r'C:\\Users\\Javier Ornelas\\OneDrive\\Documents\\HVSRdata_Main\\mHVSR Site Inv\\VSPDB Data\\CA Vertical Array Data\\HVSRdata\\2\\2.250.2\\Text_File_data\\Raw_mseed_PEG_HH'\n\n#Specify filenames of .mseed or .txt files\nh1 = 'NX.USC6..HHE.D.2022.250'\nh2 = 'NX.USC6..HHN.D.2022.250'\nv = 'NX.USC6..HHZ.D.2022.250'\n\n#specify where you want output to be stored\noutput_dir = r'C:\\Users\\Javier Ornelas\\OneDrive\\Documents\\HVSRdata_Main\\mHVSR Site Inv\\HVSR VSPDB Data'\n\nh1, h2, v, dt, time = hv.process_time_series(h1_fn=h1, h2_fn=h2, v_fn=v, directory=direc, \n                                                       file_type=1, time_cut=120)\n\n#Use the main function hv_proc which processes time series and hvsr\nwin_result, fd_select = \\\nhv.hvsr(h1=h1, h2 =h2, v=v, dt = dt, time_ts=time, output_dir=output_dir, \n                      win_width=300, overlapping=0, plot_ts=True, plot_hvsr=True, output_polar_curves=True, norm_flag = False)\n")


# # Plot the polar/azimuthal curves

# In[22]:


#Plot Polar Curve from output of hvsrProc

#specify directory where polar curve data is stored
POLAR = pd.read_csv(os.path.join(output_dir,'Test_hvsr_polar.csv'))

#specify where json file of frequencies is stored
standard_freqs = json.load(open(r'C:\Users\Javier Ornelas\OneDrive\Documents\HVSRdata_Main\Site Response Modelling Inv\GMDB OSAKA INV\mean_curves/HVSR_VSPDB_standard_frequencies.json'))

#specify degree increment
deg_increment = 10

#Specify azimuth
AZIMUTHS = list(range(0, 180, deg_increment))

#Process polar curve data
processed_pol_data = hv.process_polar_curve(polar_data = POLAR, azimuths = AZIMUTHS, standard_freqs = standard_freqs)

#Plot the polar curve
fig = hv.plot_polar_ratio(processed_pol_data)

# fig.savefig(r'C:\Users\Javier Ornelas\OneDrive\Documents\HVSRdata_Main\mHVSR Site Inv\HVSR VSPDB Data/test.png', dpi =500)


# # Plot the mean curves with the associated metadata

# In[25]:


#Specify directory where mean hvsr curve is stored
Mean_df = os.path.join(output_dir,'Test_hvsr_mean.csv')

#Specify directory where metadata is stored
metadata_csv_filepath = os.path.join(direc,'Test_metadata.csv')

#Plot the mean curve
fig = hv.plot_mean_hvsr(csv_file_path=Mean_df, metadata_csv_path=metadata_csv_filepath, xlim_m = 65, ylim_m = 4,
                              xlim=(0.1, 50), ylim=(0,8), robust_est=False, metadata_flag=True)


# # Plot the windowed time series data

# In[26]:


#Plot selected and unselected time series

#Specify directory where the time series is stored
ts_df = os.path.join(output_dir,'Test_ts_sel.csv')

#Plot the time series
fig = hv.plot_selected_time_series(csv_file_path = ts_df)

# fig.savefig(r'C:\Users\Javier Ornelas\OneDrive\Documents\HVSRdata_Main\mHVSR Site Inv\HVSR VSPDB Data/test.png', dpi =500)


# # Plot the selected HVSR curves used to develop mean curve

# In[27]:


#Plot selected time series

#Specify where the directory for the selected hvsr curves are located
sel_curve_dir = os.path.join(output_dir,'Test_hvsr_sel.csv')

#Plot the selected hvsr curves
fig = hv.plot_selected_hvsr(sel_curve_dir,xlim=(0.1, 50), ylim=(0, 8))

# fig.savefig(r'C:\Users\Javier Ornelas\OneDrive\Documents\HVSRdata_Main\mHVSR Site Inv\HVSR VSPDB Data/test.png', dpi =500)


# # Plot the mean FAS

# In[29]:


csv_path = os.path.join(output_dir,'Test_FAS_mean.csv')

fig = hv.plot_fas(csv_path, xlim=(0.01, 50), ylim=(10e3, 10e6))

# fig.savefig(r'C:\Users\Javier Ornelas\OneDrive\Documents\HVSRdata_Main\mHVSR Site Inv\HVSR VSPDB Data/test.png', dpi =500)


# # Get a csv file containing all the metadata for all stations

# In[27]:


#Example of using tools to process metadata for station xml files

#Specify the static metadata
field_crew = None
user = 'francisco javier ornelas'
weather = None
ground_type = None
sensor_ground_coupling = None
monochromatic = None
building = None
transients = None
data_type = 0
sensor_corner_frequency = 0.0083
gain = 1
comments = None

#Specify the directory where the stations are stored
directory = r'C:\Users\Javier Ornelas\OneDrive\Documents\HVSRdata_Main\mHVSR Site Inv\10'

#Run the tool
df = hmt.process_hvsr_metadata(field_crew, user, weather, ground_type, sensor_ground_coupling,
                                            monochromatic, building, transients, data_type,
                                            sensor_corner_frequency, gain, comments, directory)

#Save the dataframe (df) if needed
# df.to_csv(os.path.join(directory,'gmdb_query_metadata.csv'),header =True, index=False)

#Visualize the df
df


# # Get a csv file containing all the processing metadata for all stations

# In[19]:


#Example of using a tool to combine all metadata files

#Specify the folder path where the stations are stored
folder_path = r'C:\Users\Javier Ornelas\OneDrive\Documents\HVSRdata_Main\mHVSR Site Inv\Imperial Valley Inv'

#Specify a unique set of folders thats contained in all station folders
add_sim_path = 'Text_File_data/Raw_mseed_PEG_HH'

#Run the metadata tool
combined_metadata = hmt.combine_metadata(folder_path,add_sim_path)

#Visualize the dataframe (combined_metadata)
combined_metadata

# combined_metadata.to_csv(os.path.join(folder_path,'Imperial_Valley_processing_metadata.csv'),header =True, index=False)


# # Get a csv file containing all the mean curve data for all stations

# In[11]:


folder_path = r'C:\Users\Javier Ornelas\OneDrive\Documents\HVSRdata_Main\mHVSR Site Inv\Imperial Valley Inv'

add_sim_path = 'Text_File_data/Raw_mseed_PEG_HH'

df = hmt.create_mean_curves_csv(folder_path,'ImperialValleyMeanCurves.csv',add_sim_path, output_flag = False)

df


# In[ ]:




