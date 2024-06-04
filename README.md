# hvsrprocpy

A python library that can perform Horizontal-to-Vertical Spectral 
Ratio (HVSR) processing from recordings of microtremors or earthquakes from
seismometers. This library was developed by Francisco Javier G. Ornelas under the supervision
of Dr. Jonathan P. Stewart and Dr. Scott J. Brandenberg at University of California, Los Angeles (UCLA). 
Other contributions came from Dr. Pengfei Wang a professor at Old Dominion University, who wrote `hvsrProc` 
a rstudio package, which this python library is based on. That work can be found here:

>Wang, P. wltcwpf/hvsrProc: First release (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.4724141


# Background

HVSR is derived from ratios of the horizontal and vertical components
of a Fourier Amplitude Spectrum (FAS) from a 3-component recording of
microtremors or earthquakes. This can be done by recording ground vibrations either from
temporarily-deployed or permanently-installed seismometers, for a relatively short
period of time (~1-2 hrs) or a longer period of time.

This method or technique was first proposed by Nogoshi and Igarashi (1971) 
<a href="https://www.scirp.org/reference/referencespapers?referenceid=3100696" target="_blank">[ABSTRACT]</a> and 
later popularized by Nakamura (1989) <a href="https://trid.trb.org/View/294184" target="_blank">[ABSTRACT]</a>.
The method works by assuming that the horizontal wavefield is amplified as seismic waves propagate
through the soil deposits compared to the vertical wavefield.

# Installation

hvsrprocpy is available using pip and can be installed with:

`pip install hvsrprocpy`

# Usage

An associated example jupyter notebook should be a part this repository <a href="https://github.com/fjornelas/hvsrprocpy" target="_blank">[GIT]</a>

# Citation

If you use hvsrprocpy (directly or as a dependency of another package) for work resulting in an academic publication or
other instances, we would appreciate if you cite the following:

> Ornelas, F. J. G., Wang, P., Brandenberg, S. J., & Stewart, J. P. (2024). fjornelas/hvsrprocpy: hvsrprocpy (0.1.1). Zenodo. https://doi.org/10.5281/zenodo.11478433

# Issues

Please report any issues or leave comments on the <a href="https://github.com/fjornelas/hvsrprocpy/issues" target="_blank">Issues</a> page.

