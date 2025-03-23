
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




