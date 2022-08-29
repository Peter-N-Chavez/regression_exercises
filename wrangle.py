import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import urllib.request
from PIL import Image
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from acquire import *
from prepare import *
from explore import *
import env

def wrangle_zillow():
    url_zillow = get_db_url(env.hostname, env.username, env.password, "zillow")
    query = pd.read_sql('''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet AS cal_fin_sqf, taxvaluedollarcnt AS tax_val, yearbuilt, taxamount, fips \
                        FROM properties_2017 \
                        WHERE propertylandusetypeid = 261; ''', url_zillow)
    return query