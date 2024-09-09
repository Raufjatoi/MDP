# so i wanna create a model tha pridict the hapatitis pridiction 
# the data is like "","Category","Age","Sex","ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"
#                  "1","0=Blood Donor",32,"m",38.5,52.5,7.7,22.1,7.5,6.93,3.23,106,12.1,69
# this so lezz first create the data viz then do some eda then use some of models like 
# random forest , naive byes , knn , nn , linear regression , logistic regression and other lezz go #
# the data name is hepatitis.csv

import pandas as p 
import seaborn as s 
import matplotlib.pyplot as m 

data = p.read_csv("hepatitis.csv")