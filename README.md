<!-- Our title -->
<div align="center">
  <h3 style="font-size: 25px;">hvsrprocpy</h3>
</div>

<!-- Short description -->
<p align="center">
   A python library that performs Horizontal-to-Vertical Spectral Ratio (HVSR) processing.
</p>

<div style="text-align: center;">
    <h3 style=" font-size: 25px;">Authors</h3>
</div>

<!-- Short description -->

<p align="center">
   Francisco Javier G. Ornelas<sup>1</sup>, Pengfei Wang<sup>2</sup>, Scott J. Brandenberg<sup>1</sup>, Jonathan P. Stewart<sup>1</sup>
</p>

<sup>1</sup> University of California, Los Angeles (UCLA) <br> 
<sup>2</sup> Old Dominion University<br>


[![DOI](https://zenodo.org/badge/808826090.svg)](https://zenodo.org/doi/10.5281/zenodo.11515238)
![GitHub License](https://img.shields.io/github/license/fjornelas/hvsrprocpy)
![GitHub Release](https://img.shields.io/github/v/release/fjornelas/hvsrprocpy)

<div style="text-align: center;">
    <h3 style=" font-size: 25px;">Table of Contents</h3>
</div>

---
| Section                             | Description                                                                      |
|-------------------------------------|----------------------------------------------------------------------------------|
| [Introduction](#introduction)       | Description of the project                                                       |
| [Background](#Background)           | Background the Horizontal-to-Vertical Spectral Ratio (HVSR)                      |
| [Getting started](#Getting-started) | Description of different usages with _hvsrprocpy_ and how to install the package |
| [Citation](#Citation)               | Description of the citation for this work                                        |
| [Issues](#Issues)                   | Link showing where to report issues                                              |


# Introduction

---

A python library that can perform Horizontal-to-Vertical Spectral 
Ratio (HVSR) processing from recordings of microtremors or earthquakes from 3-component 
seismometers. This library was developed by Francisco Javier G. Ornelas under the supervision
of Dr. Jonathan P. Stewart and Dr. Scott J. Brandenberg at University of California, Los Angeles (UCLA). 
Other contributions came from Dr. Pengfei Wang a professor at Old Dominion University, who wrote `hvsrProc` 
a rstudio package, which this python library is based on. That work can be found here:

>Wang, P. wltcwpf/hvsrProc: First release (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.4724141


# Background

---

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

HVSR can be useful in site characterization since it can identify resonant frequencies at sites, through peaks in
an HVSR spectrum. Studies have also found that the lowest peak in HVSR spectra can be associated with the fundamental
frequency of a site.

# Getting started

---

## Installation


hvsrprocpy is available using pip and can be installed with:

- Jupyter Notebook
`pip install hvsrprocpy`
- PyPI
`py -m pip install hvsrprocpy`
## Usage


`hvsrprocpy` is a library that performs hvsr related processing. The library contains various features, such as:
- Manual selection of windows in the time domain and the frequency domain.
- Rotated HVSR to see the azimuthal variability of HVSR.
- Different distibutions such as normal or log-normal
- Different smoothing functions such as Konno and Ohmachi and Parzen smoothing
- Various outputs such as mean FAS and HVSR, selected HVSR and ts, and rotated HVSR.

Examples of these can be found under the examples folder in the Github repository <a href="https://github.com/fjornelas/hvsrprocpy" target="_blank">[GIT]</a>

### Example of manual window selection on time series

<img src="https://github.com/fjornelas/hvsrprocpy/blob/main/figs/microtremor_ts_win_sel_example.png?raw=true" width="775">

### Example of manual  window selection on HVSR

<img src="https://github.com/fjornelas/hvsrprocpy/blob/main/figs/hvsr_fas_win_sel_example.png?raw=true" width="775">

### Example of azimuthal plots

<img src="https://github.com/fjornelas/hvsrprocpy/blob/main/figs/hvsr_polar_example.png?raw=true" width="775">

### Example of Mean Curve plot with Metadata

<img src="https://github.com/fjornelas/hvsrprocpy/blob/main/figs/hvsr_mean_curve_and_meta_example.png?raw=true" width="775">

### Example comparisons of Konno and Ohmachi and Parzen smoothing on FAS

<div style="display: flex; flex-direction: row;">
    <img src="https://github.com/fjornelas/hvsrprocpy/blob/main/figs/fas_ko_smoothing_example.png?raw=true" width="400" style="margin-right: 10px;">
    <img src="https://github.com/fjornelas/hvsrprocpy/blob/main/figs/fas_parzen_smoothing_example.png?raw=true" width="400">
</div>


# Citation

---

If you use hvsrprocpy (directly or as a dependency of another package) for work resulting in an academic publication or
other instances, we would appreciate if you cite the following:

> Ornelas, F. J. G., Wang, P., Brandenberg, S. J., & Stewart, J. P. (2024). fjornelas/hvsrprocpy: hvsrprocpy (0.1.1). Zenodo. https://doi.org/10.5281/zenodo.11478433

# Issues

---

Please report any issues or leave comments on the <a href="https://github.com/fjornelas/hvsrprocpy/issues" target="_blank">Issues</a> page.

## License

This project has been licensed under [![The GNU General Public License v3.0](https://www.gnu.org/graphics/gplv3-88x31.png "The GNU General Public License v3.0")](https://www.gnu.org/licenses/gpl-3.0.en.html)
more information about the license can be found here <a href="https://github.com/fjornelas/hvsrprocpy/blob/0.1.0/LICENSE" target="_blank">[LICENSE]</a>

## Acknowledgements

We would also like to thank the many others who aided in the development of this python library, these are:

