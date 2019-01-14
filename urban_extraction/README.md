# Urban extraction using Support Vector Machine classifier

## code structure

The code for performing urban extraction is located into the `classification_module` folder. The stracture of this folder is as follows:

- `classifier:` It contains the SVM classifier for performing the urban extraction (additional classifiers can be added into this folder).

NOTE:  An issue of using snappy is the performance. Unfortunately, the performance of snappy compared to SNAP desktop 
       is considerably slower. SNAP has been set up to execute tasks using parallel processing unlike snappy.

       reference: https://forum.step.esa.int/t/performance-of-snap-desktop-and-snappy/1850/9

       As an alternative, the bash script `texture.sh` can be used. This script uses the Graphicl Processing Tool (GPT)
       which is a commnad line tool to access SNAP all funtionalities. It uses multithreding and the performance is the
       same with SNAP desktop 

- `features:` It contains features that can be used as additional layers to improve the accuracy of urban extraction. At the moment, only texture analysis (GLCM) is implemeted.

- `indices:` It contains vegetation(NDVI) and water(NDWI) index. NDVI index is useful for masking out the vegetation. In that way, the accuracy of urban extraction is further improved


## run the code

`cd classification_module/classifier`
Run SVM classifier
`python SVM.py` 
Run SVM classifier with texture analysis
`python SVM_texture.py`
