## [ Deep-learning-derived planetary boundary layer height from conventional meteorological measurements (2024) ](../papers/acp-24-6477-2024.pdf)

---

### Introduction

what is the Planetary Boundary Layer (PBL)?
- atmosphere's lowest part
- directly influences meterological variables
- (PBLH) height

what is SONDE?
- radiosonde
- standard measurement methods for PBLH

what is AERI?kv
- Atmospheric Emitted Radiance Interferometer

deep neural networks used in this approach

ensemble DNN with multi-structure design

### Data and Instruments

#### ARM sites

what is atmospheric radiation measurement (ARM)?
- datasets of weather data

more data gathering technologies and techniques


#### Deep learning model to estimate PBLH

##### The multi-structure deep learning model

seems like a typical deep learning setup with some tweaks to the overall structure

using tensorflow
comprehensive model, filters to make sure only impactful data guides the model

what is curtail overfitting?

what is the Adam optimizer?
- some loss function

### Training the DNN model

trained using a PBLH dataset enriched by SONDE and lidar measurements during 1994 - 2016 over the SGP.

subset of dta from 2017 - 2020 is used for testing of the prediction results


### Summary
Region specific refinement to the model is necessary

## [ A meteorological analysis interpolation scheme for high spatial-temporal resolution in complex terrain (2020) ](../papers/Clustering_TempRH_Downscale20.pdf)

---

### Introduction

An adaptive high-temporal resolution interpolation scheme for meterological observations is presented.

