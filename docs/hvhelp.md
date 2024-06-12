# _hvhelp_ (HVSR Helper Tools)

---

This file contains a set of functions that 
can be used to process microtremor or earthquake files in
.mseed or .txt file format.

The definitions of these functions are below:

## _proc_mseed_data_

Function to process .mseed files.

`proc_mseed_data(file_direc, h1_fn, h2_fn, v_fn, trim_flag=False, time_cut=300)`

`Parameters:`
    
**file_direc** (string) - Specifies where the files are stored.

**h1_fn** (string) - Specifies the name of the file for the h1 component.

**h2_fn** (string) - Specifies the name of the file for the h2 component.

**v_fn** (string) - Specifies the name of the file for the v component.

**trim_flag** (boolean) - Specifies whether to trim the trace. Default = False.

**time_cut** (int) - Specifies how much time cut in seconds. Default = 300
    
`returns:` 

An array of each component.

## _proc_txt_data_

Function to process .txt or .csv files.

`proc_txt_data(file_direc, h1_fn, h2_fn, v_fn, time_cut=300)`

`Parameters:`
    
**file_direc** (string) - Specifies where the files are stored.

**h1_fn** (string) - Specifies the name of the file for the h1 component.

**h2_fn** (string) - Specifies the name of the file for the h2 component.

**v_fn** (string) - Specifies the name of the file for the v component.

**time_cut** (int) - Specifies how much time cut in seconds. Default = 300
    
`returns:` 

An array of each component.

## _process_time_series_

Function processes time series files in .txt format or .mseed format.

`process_time_series(h1_fn, h2_fn, v_fn, directory, **kwargs)`

`Parameters:`

**file_direc** (string) - Specifies where the files are stored.

**h1_fn** (string) - Specifies the name of the file for the h1 component.

**h2_fn** (string) - Specifies the name of the file for the h2 component.

**v_fn** (string) - Specifies the name of the file for the v component.

`kwargs:`

**time_cut** (int) - Integer representing how much time to be cut from the beginning
and end of the amplitude component of a  time series array.  

**file_type** (int) - Indicates whether file is of mseed or txt file type. Default = 1 (mseed) 

`returns:`

An array of each component