
# Work Notes
###### Kentucky Mesonet Spring Machine Learning Efforts

#### Contained here is the ml-soundings-archive-main
This dir is the codebase of the Aies paper discussing data with ml

Install Pymica

#### [ MOOC Course ](https://learning.ecmwf.int/course/view.php?id=46)

#### [ AMS ML Course ](https://annual.ametsoc.org/index.cfm/2025/your-annual/registration/short-course-registration/)

#### [ CIRA Course ](https://www.cira.colostate.edu/ml/home/)


# General Info

Subset KY points
- Spatial: -89.813 -81.688
            36.188  39.438

## Tue Feb 4

Focus on pymica, specifically how to format our mesonet data into their codes.
Start going through MOOC course
 - I have already done the first module for this as of writing

---

## Mon Feb 17

- Update climate codes to receive input data as formatted
- format data for pymica
- continue the MOOC

## Tue Feb 18 

- subset the PRISM data for KY using the 4 lat long data points
- format that data for the input of pymica
- tmean and prcp

## Thu Feb 27

- Get timestamps in the csv converted files
- PRCP and TMEAN are the data points to work with 

## Fri Feb 28

### [ pymica examples ](https://pymica.readthedocs.io/en/latest/01_howto_prepare_data.html)

- Look over pymica data preperation files, look over csv file format
- How do we tranform our data to match their input?
- 1200km for the distance (in csv file)

- Look at polars instead of pandas

## Mon Mar 17

### Kelcee
- Scikit learn python package
    - Random forest regression 
    - XGBoost regression
- Tensor Flow if time allows

### Cy
- complete code to run through all the data

## Mon Mar 21

### Cy

- subset data for KY BEFORE data formatting
- climate change indicies code to eric as well, in python

#### commands for codes

python ky_prism_data_proccessor.py /Volumes/PRISMdata/PRISM_data/an/tmean/daily/2019 --output /Volumes/PRISMdata/pymica_PRISM_data/ --pattern 2019  --subsample 1 --dem_file /Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat
python threaded_ky_processor.py /Volumes/PRISMdata/PRISM_data/an/tmean/daily/2019 --output /Volumes/PRISMdata/pymica_PRISM_data/ --pattern 2019  --subsample 1 --dem_file /Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat
python multiprocess_ky_processor.py /Volumes/PRISMdata/PRISM_data/an --variable tmean --output_dir /Volumes/PRISMdata/pymica_PRISM_data/ --subsample 1 --dem_file /Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat --processes 5 --elevation_threads 2

## Mon Mar 31

## April 11

Review literature on machine learning algoriths and inter[olationg materolgical observations on hig-res Cartisien grids. 

### Findings
- https://link.springer.com/article/10.1007/s00704-016-2003-7
- https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.8641
- https://iopscience.iop.org/article/10.1088/2632-2153/ad4b94/meta

### Keywords

## April 29

### Kelcee
- Feature extraction
- look over machine learning notes in google drive (Lecture 2 - feature extraction)

