# -*- coding: utf-8 -*-
"""
@author: 
    Nilimesh Halder, PhD
    BSc in Computer Science and Engineering 
        @ Khulna University.
    PhD in Artificial Intelligence (Modelling & Simulation) 
        @ The University of Western Australia.
"""

# ----------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Load necessary libraries
import time
import sqlalchemy as sa
import pandas as pd
import pickle as pk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from pandas.plotting import scatter_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ----------------------------------------------------------------------
# Load DataSet from CSV file
def loadFrCSVFile(filename):
    print('Data File Name: {}'.format(filename))
    col_names = [
       'SampleCodeNumber', 'ClumpThickness', 
       'CellSize', 'CellShape', 'MarginalAdhesion', 
       'EpithelialCellSize', 'BareNuclei', 'BlandChromatin', 
       'NormalNucleoli', 'Mitoses', 'Class']
    dataset = pd.read_csv(filename, names=col_names)
    return dataset
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------    
# Import DataSet to a MySQL Database
def import2MySQL(dataset):
    engine_str = (
      'mysql+pymysql://{user}:{password}@{server}/{database}'.format(
       user      =  'root',
       password  =  'root888',
       server    =  'localhost',
       database  =  'datasciencerecipes'))
    engine = sa.create_engine(engine_str)
    conn = engine.connect()

    # check whether connection is Successful or not
    if (conn): 
        print("MySQL Connection is Successful ... ... ...")    
    else:  
        print("MySQL Connection is not Successful ... ... ...")

    dataset.to_sql(name='breastcancerdata', con=engine, 
                   schema='datasciencerecipes', 
                   if_exists = 'replace', chunksize = 1000, 
                   index=False)
    conn.close()

# ----------------------------------------------------------------------    
# Load DataSet from MySQL Database to Pandas a DataFrame
def loadDataSetFrMySQLTable():
    engine_str = (
      'mysql+pymysql://{user}:{password}@{server}/{database}'.format(
       user      =  'root',
       password  =  'root888',
       server    =  'localhost',
       database  =  'datasciencerecipes'))
    
    engine = sa.create_engine(engine_str)
    conn = engine.connect()
    
    # check whether connection is Successful or not
    if (conn): 
        print("MySQL Connection is Successful ... ... ...")    
    else:      
        print("MySQL Connection is not Successful ... ... ...")

    # MySQL Query with few generated Attributes/Features
    query = '''
    SELECT  
        SampleCodeNumber, ClumpThickness, 
        CellSize, CellShape, MarginalAdhesion, 
        EpithelialCellSize, BareNuclei, BlandChromatin, 
        NormalNucleoli, Mitoses, Class 
    FROM breastcancerdata;
    '''
    query_result = conn.execute(query)
    dataset =  pd.DataFrame(query_result.fetchall(), 
                            columns =  query_result.keys())
    print('DataFrame Size',dataset.shape)
    print('ROW',dataset.shape[0])
    print('COLUMN',dataset.shape[1])
    conn.close()

    # drop 'SampleCodeNumber' from the training dataset
    dataset = dataset.drop(['SampleCodeNumber'], axis = 1)
    
    print('\n')    
    # Get Information on the Dataset
    print(dataset.info())
    
    # Count Number of Missing Value on Each Column    
    print('\nCount Number of Missing Value on Each Column: ')        
    print(dataset.isnull().sum(axis=0))
    # Count Number of Missing Value on Each Row    
    #print('\nCount Number of Missing Value on Each Row: ')        
    #print(dataset.isnull().sum(axis=1))

    # --------------------------------------------------
    # Missing Values treatment of the DataSet (if any)
    # --------------------------------------------------    
    # a) Filling NULL values with Zeros
    #dataset = dataset.fillna(0)
    #print('\nCount Number of Missing Value on Each Column: ')        
    ## Count Number of Missing Value on Each Column
    #print(dataset.isnull().sum(axis=0))
    #print('\nCount Number of Missing Value on Each Row: ')        
    ## Count Number of Missing Value on Each Row
    #print(dataset.isnull().sum(axis=1))

    # b) Filling NULL values according to their dataTypes
    # Group Dataset according to different dataTypes
    gd = dataset.columns.to_series().groupby(dataset.dtypes).groups
    print('\nGroup Columns according to their dataTypes: \n', gd)  
    colNames = dataset.columns.values.tolist()
    
    #dataset[['Class']] = dataset[['Class']].astype('string')
    #dataset[['BareNuclei']] = dataset[['BareNuclei']].astype('int64')
    
    for colName in colNames:
        if dataset[colName].dtypes == 'int64':
           dataset[colName] = dataset[colName].fillna(0)
        if dataset[colName].dtypes == 'float64':
           dataset[colName] = dataset[colName].fillna(0.0) 
        if dataset[colName].dtypes == 'object':
           dataset[colName] = dataset[colName].fillna('Unknown') 
    
    ## Count Number of Missing Value on Each Column
    #print('\nCount Number of Missing Value on Each Column: ')        
    #print(dataset.isnull().sum(axis=0))
    ## Count Number of Missing Value on Each Row    
    #print('\nCount Number of Missing Value on Each Row: ')        
    #print(dataset.isnull().sum(axis=1))

    #dataset.ix[:, 'NumericalClass'] = 0
    #dataset.ix[dataset.Class == 'M', 'NumericalClass'] = 1
    #dataset.ix[dataset.Class == 'R', 'NumericalClass'] = 0
    
    print('COLUMN',dataset.shape[1])

    return dataset

# Helper modules for Descriptive Statistics
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# Helper modules for Descriptive Statistics
def get_top_abs_correlations(df, n=5): 
    #au_corr = df.corr().abs().unstack()
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(
                                                   ascending=False
                                                   )
    return au_corr[0:n]

# Helper modules for Descriptive Statistics
def corrank(X):
    import itertools
    df = pd.DataFrame([[(i,j), 
         X.corr().loc[i,j]] for i,j in list(itertools.combinations(
                                                X.corr(), 
                                                2))],
         columns=['pairs','corr'])
    print(df.sort_values(by='corr',ascending=False))
    print()   

# ----------------------------------------------------------------------
# Data Summarisation to understand datasets USING Descriptive Statistics
# ----------------------------------------------------------------------
def summariseDataset(dataset):

    # Separate out Numerical & Categorical Variables
    colNumeric = []; colCategory = []; target = 'Class';
    colNames = dataset.columns.values.tolist()
    for colName in colNames:
        if dataset[colName].dtypes == 'int64':
           colNumeric.append(colName)
        if dataset[colName].dtypes == 'float64':
           colNumeric.append(colName)
        if dataset[colName].dtypes == 'object':
           colCategory.append(colName)

    print()           
    print('Number of Categocial Features: ', len(colCategory))
    print('Number of Numerical Features: ', len(colNumeric))
    print('Target Column Name: ', target)
    print()
    
    #removing target variable from the column lists
    #colCategory.remove('Class'); 
    colNumeric.remove('Class');
    
    # ------------------------------------------------    
    # Descriptive statistics: Numerical Columns
    # ------------------------------------------------    
    # shape
    print(dataset[colNumeric].shape)
    # types
    print(dataset[colNumeric].dtypes)
    # head
    print(dataset[colNumeric].head(5))
    # descriptions
    print(dataset[colNumeric].describe())

    # descriptions, change precision to 2 places
    pd.set_option('precision', 1)
    print(dataset[colNumeric].describe())
    # correlation
    pd.set_option('precision', 2)
    print(dataset[colNumeric].corr())    
    
    # Ranking of Correlation Coefficients among Variable Pairs
    print()
    print("Ranking of Correlation Coefficients:")    
    corrank(dataset[colNumeric])

    # Print Highly Correlated Variables
    print()
    print("Highly correlated variables (Absolute Correlations):")
    print(get_top_abs_correlations(dataset[colNumeric], 15))

    # ------------------------------------------------    
    # Target distribution
    # ------------------------------------------------    
    print("\n\nTarget distribution:")
    print(dataset['Class'].describe())

    # ------------------------------------------------
    # Descriptive statistics: Categorical Columns
    # ------------------------------------------------
    # Category distribution
    print()    
    print('Distribution of Categorical Variables: ')
    for colName in colCategory:
        print()
        print(dataset.groupby(colName).size())

# ----------------------------------------------------------------------
# Data Visualisation to understand & visualise Datasets
# ----------------------------------------------------------------------    
def visualiseDataset(dataset):
    print()           
    print('Data Visualisation Part ... ... ...')
    
    # Separate out Numerical & Categorical Variables
    colNumeric = []; colCategory = []; target = 'Class';
    colNames = dataset.columns.values.tolist()

    for colName in colNames:
        if dataset[colName].dtypes == 'int64':
           colNumeric.append(colName)
        if dataset[colName].dtypes == 'float64':
           colNumeric.append(colName)
        if dataset[colName].dtypes == 'object':
           colCategory.append(colName)

    print()           
    print('Number of Categocial Features: ', len(colCategory))
    print('Number of Numerical Features: ', len(colNumeric))
    print('Target Column Name: ', target)
    print()

    #remove target variable from the column lists
    #colCategory.remove('Class'); 
    colNumeric.remove('Class');

    # --------------------------------------
    # Data visualizations
    # --------------------------------------
    # BOX plots USING box and whisker plots
    i = j =  1
    print()
    print('BOX plot of each numerical features')
    pyplot.figure(figsize=(11,9))     
    for col in colNumeric:
        plt.subplot(7,6,i)
        plt.axis('on')
        plt.tick_params(axis='both', left='on', top='off', 
            right='off', bottom='on', labelleft='off', labeltop='off', 
                        labelright='off', labelbottom='off')
        dataset[col].plot(kind='box', subplots=True,  
                        sharex=False, sharey=False)
        i += 1
    pyplot.show()
        
    # USING histograms
    print()
    print('Histogram of each Numerical Feature')
    pyplot.figure(figsize=(11,9))     
    for col in colNumeric:
        plt.subplot(7,6,j)
        plt.axis('on')
        plt.tick_params(axis='both', left='on', top='off', 
            right='off', bottom='on', labelleft='off', labeltop='off', 
                        labelright='off', labelbottom='off')
        
        dataset[col].hist()
        j += 1
    pyplot.show()

    # USING scatter plot matrix
    j = 1
    print()
    print('Scatter Plot of each Numerical Feature vs Target') 
    pyplot.figure(figsize=(11,9))     
    for col in colNumeric:
        plt.subplot(7,6,j)
        plt.axis('on')
        plt.tick_params(axis='both', left='on', top='off', 
            right='off', bottom='on', labelleft='off', labeltop='off', 
                        labelright='off', labelbottom='off')
        plt.scatter(dataset[col], dataset['Class']) #target<-SaleProce 
        j += 1        
    pyplot.show()

    # USING density plots
    j = 1
    print()
    print('Density Plot of each Numerical Feature') 
    pyplot.figure(figsize=(11,9))     
    for col in colNumeric:
        plt.subplot(7,6,j)
        plt.axis('on')
        plt.tick_params(axis='both', left='on', top='off', 
            right='off', bottom='on', labelleft='off', labeltop='off', 
                        labelright='off', labelbottom='off')
        dataset[col].plot(kind='density', subplots=True,  
                        sharex=False, sharey=False)
        j += 1        
    pyplot.show()

    # correlation matrix
    print()
    print('Correlation Matrix of All Numerical Features')   
    fig = pyplot.figure(figsize=(11,9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataset[colNumeric].corr(), 
                     vmin=-1, vmax=1, 
                     interpolation='none')
    fig.colorbar(cax)
    ticks = np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    #ax.set_xticklabels(colNumeric)
    #ax.set_yticklabels(colNumeric)
    pyplot.show()

    # ---------------------------------------
    # Correlation Plot using seaborn
    # ---------------------------------------
    print()
    print("Correlation plot of Numerical features")
    # Compute the correlation matrix
    corr = dataset[colNumeric].corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin= -1.0, 
                center=0, square=True, linewidths=.5, 
                cbar_kws={"shrink": .5})
    pyplot.show()

    # ---------------------------------------
    # Pie chart for Categorical Variables
    # ---------------------------------------
    # Category distribution
    print()    
    print('PIE Chart of each Categorical Variable: ')
    pyplot.figure(figsize=(11,9)) 
    i =1
    for colName in colCategory:
        labels = []; sizes = [];
        df = dataset.groupby(colName).size()
        for key in df.keys():
            labels.append(key)
            sizes.append(df[key])

        # Plot PIE Chart with %
        #print()
        #print(colName)
        plt.subplot(9,5,i)
        plt.axis('on')
        plt.tick_params(axis='both', left='off', top='off', 
            right='off', bottom='off', labelleft='off', 
            labeltop='off', labelright='off', labelbottom='off')        
        
        plt.pie(sizes, #labels=labels, 
                #autopct='%1.1f%%', 
                shadow=True, startangle=140)
        plt.axis('equal')
        i += 1
    #plt.savefig('Piefig.pdf', format='pdf')
    plt.show()
    
# Helper module for Label Encoding for Categorical Features
def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category',
                                                     'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

# ----------------------------------------------------------------------
# Data Pre-Processing
# ----------------------------------------------------------------------    
def preProcessingData(dataset):
    # 1. Data Cleaning
    
    # Separate out Numerical & Categorical Variables
    colNumeric = []; colCategory = []; target = 'Class';
    colNames = dataset.columns.values.tolist()

    # Convert Year (numerical) Cols to Categorical Cols
    #dataset[['Class']] = dataset[['Class']].astype('object')

    for colName in colNames:
        if dataset[colName].dtypes == 'int64':
           colNumeric.append(colName)
        if dataset[colName].dtypes == 'float64':
           colNumeric.append(colName)
        if dataset[colName].dtypes == 'object':
           colCategory.append(colName)
    print() 
    print('Number of Categocial & Numerical Features: ', 
                                          len(colNames))          
    print('Number of Categocial Features: ', len(colCategory))
    print('Number of Numerical Features: ', len(colNumeric))
    print('Target Column Name: ', target)  

  
    # --------------------------
    # Only Numerical Features
    # --------------------------    
    cols_X = colNumeric     
    cols_X.remove('Class')
    print(cols_X)

    # ---------------------------------------    
    # ALL Numerical & Categorical Features with Encoding
    # ---------------------------------------    
    #cols_X = colNames; 
    #cols_X.remove('SalePrice'); cols_X.remove('ClassLabel');
    #cols_X.remove('NumericalClass');    
    #dataset = dummyEncode(dataset.loc[:, cols_X])    
    
    cols_Y = target; #print(cols_Y)

    # 2. Data Transform - Split out train : test datasets
    train_X, test_X, train_Y, test_Y = train_test_split(
                                            dataset.loc[:, cols_X], 
                                            dataset.loc[:, cols_Y], 
                                            test_size=0.33,
                                            )
    return train_X, test_X, train_Y, test_Y

# ----------------------------------------------------------------------
# Applied Machine Learning Algorithm ... ... ...
# ----------------------------------------------------------------------    
def evaluateAlgorithm(train_X, test_X, train_Y, test_Y, dataset):
    # Evaluate Machine Lreaning Algorithm, Parameter settings etc.
    # Algorithms are applied with different parameter settings
    # manual parameter tuning is shown here
    # Grid & Random Search for Parameters tuning will be shown later
    
    KFold = 3
    model_List = []
    cv_outcomes = []
    description = []
    randomSeed = 7    

    #####################################################################
    # Neural Networks Algorithms ########################################
    #####################################################################
    # -------------------------------------------------------------------
    # Explore Neural Network Algorithm from Scikit Learn
    # -------------------------------------------------------------------
    # Neural Network i.e. MLPClassifier() 
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # PS - Parameter Settings - (manual tuning)
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------    
    # Algorithm initialisation - PS - 1   
    MLP_1 = MLPClassifier(activation = 'tanh', solver = 'adam', 
            hidden_layer_sizes = 100,
            alpha=0.0001, batch_size='auto', 
            learning_rate='constant', learning_rate_init=0.001, 
            power_t=0.5, max_iter=200, shuffle=True, 
            random_state=None, tol=0.0001, verbose=False, 
            warm_start=False, momentum=0.9, 
            nesterovs_momentum=True, early_stopping=False, 
            validation_fraction=0.1, beta_1=0.9, 
            beta_2=0.999, epsilon=1e-08) 
    model = MLP_1
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('MLP_1', model)])                    
    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('ML_1')
    
    # Cross Validation Results
    print("\n%s: " % ('Neural Network Algorithm: PS-1'))
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('MLP_1', 'Neural Network Algorithm: PS-1', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))   


    # -------------------------------------------------------------------
    # Algorithm initialisation - PS - 2   
    MLP_2 = MLPClassifier(activation = 'relu', solver = 'sgd', 
            hidden_layer_sizes = 150,
            alpha=0.0001, batch_size='auto', 
            learning_rate='constant', learning_rate_init=0.001, 
            power_t=0.5, max_iter=200, shuffle=True, 
            random_state=None, tol=0.0001, verbose=False, 
            warm_start=False, momentum=0.9, 
            nesterovs_momentum=True, early_stopping=False, 
            validation_fraction=0.1, beta_1=0.9, 
            beta_2=0.999, epsilon=1e-08) 
    model = MLP_2
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('MLP_2', model)])     
    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('ML_2')
    
    # Cross Validation Results
    print("\n%s: " % ('Neural Network Algorithm: PS-2'))
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('MLP_2', 'Neural Network Algorithm: PS-2', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))   


    # -------------------------------------------------------------------    
    # Algorithm initialisation - PS - 3   
    MLP_3 = MLPClassifier(activation = 'tanh', solver = 'lbfgs', 
            hidden_layer_sizes = 100,
            alpha=0.0001, batch_size='auto', 
            learning_rate='constant', learning_rate_init=0.001, 
            power_t=0.5, max_iter=200, shuffle=True, 
            random_state=None, tol=0.0001, verbose=False, 
            warm_start=False, momentum=0.9, 
            nesterovs_momentum=True, early_stopping=False, 
            validation_fraction=0.1, beta_1=0.9, 
            beta_2=0.999, epsilon=1e-08) 
    model = MLP_3
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('MLP_3', model)])     
    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('ML_3')
    
    # Cross Validation Results
    print("\n%s: " % ('Neural Network Algorithm: PS-3'))
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('MLP_3', 'Neural Network Algorithm: PS-3', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))  


    # -------------------------------------------------------------------
    # PS-Parameter Settings - (automatic tuning) using GridSearchCV()
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Algorithm initialisation - PS - 4 
    model = MLPClassifier()
    parameters = {'activation' : ['relu','tanh'],
                  'solver' : ['sgd', 'adam'],
                  'learning_rate' : ['constant', 
                                     'adaptive'],
                  'hidden_layer_sizes' : [150,200],
                  'max_iter': [350,450]
                  # Add more parameters here for tuning
                 }
    grid = GridSearchCV(estimator=model, param_grid = parameters,
                        cv = KFold, verbose = 1, n_jobs = 4)
    grid.fit(train_X, train_Y)

    # Results from Grid Search
    print("\n========================================================")
    print(" Results from Grid Search " )
    print("========================================================")    
    print("\n The best estimator across ALL searched params:\n",
          grid.best_estimator_)
    print("\n The best score across ALL searched params:\n",
          grid.best_score_)
    print("\n The best parameters across ALL searched params:\n",
          grid.best_params_)
    print("\n ========================================================")

    # --------------------------------------
    # Model setup using grid search results
    # --------------------------------------
    model = grid.best_estimator_
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('MLP_4', model)])     

    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 1)
    cv_outcomes.append(cv_results)
    description.append('MLP_4')
    
    # Cross Validation Results
    print("\n%s: " % ('Neural Network Algorithm: PS-4'))    
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('MLP_4', 'Neural Network Algorithm: PS-4', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))    

    # -------------------------------------------------------------------
    # PS-Parameter Settings - (automatic tuning) using RandomSearchCV()
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Algorithm initialisation - PS - 5 
    model = MLPClassifier()
    parameters = {
        'activation' : ['relu','tanh'],
        'solver' : ['sgd', 'adam'],
        'learning_rate' : ['constant', 'adaptive'],
        'hidden_layer_sizes' : sp_randInt(100, 200),
        'max_iter': sp_randInt(100, 500),
        'learning_rate_init' : sp_randFloat()
        # Add more parameters here for tuning
                 }    
    randm = RandomizedSearchCV(estimator=model, 
                        param_distributions = parameters,
                        cv = KFold, n_iter = 10, verbose = 1,
                        n_jobs = 4)
    randm.fit(train_X, train_Y)

    # Results from Random Search
    print("\n========================================================")
    print(" Results from Random Search " )
    print("========================================================")    
    print("\n The best estimator across ALL searched params:\n",
          randm.best_estimator_)
    print("\n The best score across ALL searched params:\n",
          randm.best_score_)
    print("\n The best parameters across ALL searched params:\n",
          randm.best_params_)
    print("\n ========================================================")

    # ----------------------------------------
    # Model setup using random search results
    # ----------------------------------------
    model = randm.best_estimator_
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('MLP_5', model)])     

    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 1)
    cv_outcomes.append(cv_results)
    description.append('MLP_5')
    
    # Cross Validation Results
    print("\n%s: " % ('Neural Network Algorithm: PS-5'))    
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('MLP_5', 'Neural Network Algorithm: PS-5', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix)) 

    #####################################################################
    # NB, KNN, SVM Algorithms ########################################
    #####################################################################
    # -------------------------------------------------------------------
    # Explore Naive Bayes Algorithm from Scikit Learn
    # -------------------------------------------------------------------
    # Gaussian Naive Bayes i.e. GaussianNB() 
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # PS - Parameter Settings - (manual tuning)
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------    
    # Algorithm initialisation - PS - 1   
    NB_1 = GaussianNB() 
    model = NB_1
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('NB_1', model)])     
    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('NB_1')
    
    # Cross Validation Results
    print("\n%s: " % ('GaussianNB Algorithm: PS-1'))
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('NB_1', 'GaussianNB Algorithm: PS-1', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))   

    # -------------------------------------------------------------------    
    # Algorithm initialisation - PS - 2   
    NB_2 = MultinomialNB() 
    model = NB_2
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('NB_2', model)])      
    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('NB_2')
    
    # Cross Validation Results
    print("\n%s: " % ('MultinomialNB Algorithm: PS-2'))
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('NB_2', 'MultinomialNB Algorithm: PS-2', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))   

    # -------------------------------------------------------------------    
    # Algorithm initialisation - PS - 3   
    NB_3 = BernoulliNB() 
    model = NB_3
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('NB_3', model)])      
    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('NB_3')
    
    # Cross Validation Results
    print("\n%s: " % ('BernoulliNB Algorithm: PS-3'))
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('NB_3', 'BernoulliNB Algorithm: PS-3', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))   

    # -------------------------------------------------------------------
    # Explore KNN Algorithm from Scikit Learn
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # PS - Parameter Settings - (manual tuning)
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------    
    # Algorithm initialisation - PS - 1   
    KNN_1 = KNeighborsClassifier(n_neighbors=5, 
            weights='uniform', algorithm='auto', leaf_size=30, 
            p=2, metric='minkowski', metric_params=None, 
            n_jobs=4)
    model = KNN_1
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('KNN_1', model)])      
    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('KNN_1')
    
    # Cross Validation Results
    print("\n%s: " % ('KNN Algorithm: PS-1'))
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('KNN_1', 'KNN Algorithm: PS-1', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))   

    # -------------------------------------------------------------------
    # PS-Parameter Settings - (automatic tuning) using GridSearchCV()
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Algorithm initialisation - PS - 2 
    model = KNeighborsClassifier()
    parameters = {'algorithm' : ['auto','ball_tree','kd_tree','brute'],
                  'weights' : ['uniform', 'distance'],
                  'n_neighbors' : [5,10,15,20],
                  'leaf_size' : [30,45,60],
                  'p' : [1,2]
                  # Add more parameters here for tuning
                 }
    grid = GridSearchCV(estimator=model, param_grid = parameters,
                        cv = KFold, verbose = 0, n_jobs = 4)
    grid.fit(train_X, train_Y)

    # Results from Grid Search
    print("\n========================================================")
    print(" Results from Grid Search " )
    print("========================================================")    
    print("\n The best estimator across ALL searched params:\n",
          grid.best_estimator_)
    print("\n The best score across ALL searched params:\n",
          grid.best_score_)
    print("\n The best parameters across ALL searched params:\n",
          grid.best_params_)
    print("\n ========================================================")

    # --------------------------------------
    # Model setup using grid search results
    # --------------------------------------
    model = grid.best_estimator_
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('KNN_2', model)])     

    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('KNN_2')
    
    # Cross Validation Results
    print("\n%s: " % ('KNN Algorithm: PS-2'))    
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('KNN_2', 'KNN Algorithm: PS-2', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix)) 

    # -------------------------------------------------------------------
    # PS-Parameter Settings - (automatic tuning) using RandomSearchCV()
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Algorithm initialisation - PS - 3
    model = KNeighborsClassifier()
    parameters = {'algorithm' : ['auto','ball_tree',
                                 'kd_tree','brute'],
                  'weights' : ['uniform', 'distance'],
                  'n_neighbors' : sp_randInt(5, 50),
                  'leaf_size' : sp_randInt(10, 100),
                  'p' : [1,2]
                  # Add more parameters here for tuning
                 }    
    
    randm = RandomizedSearchCV(estimator=model, 
                        param_distributions = parameters,
                        cv = KFold, n_iter = 10, verbose = 0,
                        n_jobs = 4)
    randm.fit(train_X, train_Y)

    # Results from Random Search
    print("\n========================================================")
    print(" Results from Random Search " )
    print("========================================================")    
    print("\n The best estimator across ALL searched params:\n",
          randm.best_estimator_)
    print("\n The best score across ALL searched params:\n",
          randm.best_score_)
    print("\n The best parameters across ALL searched params:\n",
          randm.best_params_)
    print("\n ========================================================")

    # ----------------------------------------
    # Model setup using random search results
    # ----------------------------------------
    model = randm.best_estimator_
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('KNN_3', model)])     

    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('KNN_3')
    
    # Cross Validation Results
    print("\n%s: " % ('KNN Algorithm: PS-3'))    
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('KNN_3', 'KNN Algorithm: PS-3', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))

    # -------------------------------------------------------------------
    # Explore SVM Algorithm from Scikit Learn
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # PS - Parameter Settings - (manual tuning)
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------    
    # Algorithm initialisation - PS - 1   
    SVC_1 = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', 
            coef0=0.0, shrinking=True, probability=False, 
            tol=0.001, cache_size=200, class_weight=None, 
            verbose=False, max_iter=-1, 
            decision_function_shape='ovr', 
            random_state=randomSeed)
    model = SVC_1
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('SVC_1', model)])     
    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('SVC_1')
    
    # Cross Validation Results
    print("\n%s: " % ('SVM Algorithm: PS-1'))
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('SVC_1', 'SVC Algorithm: PS-1', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))   

    # -------------------------------------------------------------------
    # PS-Parameter Settings - (automatic tuning) using GridSearchCV()
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Algorithm initialisation - PS - 2 
    model = SVC()
    parameters = {'C' : [0.5,0.3,0.9],
                  'kernel' : ['linear', 'poly', 'rbf',
                              'sigmoid'],
                  'gamma' : ['auto'],
                  'cache_size' : [200,300,400],
                  # Add more parameters here for tuning
                 }
    grid = GridSearchCV(estimator=model, param_grid = parameters,
                        cv = KFold, verbose = 0, n_jobs = 4)
    grid.fit(train_X, train_Y)

    # Results from Grid Search
    print("\n========================================================")
    print(" Results from Grid Search " )
    print("========================================================")    
    print("\n The best estimator across ALL searched params:\n",
          grid.best_estimator_)
    print("\n The best score across ALL searched params:\n",
          grid.best_score_)
    print("\n The best parameters across ALL searched params:\n",
          grid.best_params_)
    print("\n ========================================================")

    # --------------------------------------
    # Model setup using grid search results
    # --------------------------------------
    model = grid.best_estimator_
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('SVC_2', model)])     

    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('SVC_2')
    
    # Cross Validation Results
    print("\n%s: " % ('SVM Algorithm: PS-2'))    
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('SVC_2', 'SVC Algorithm: PS-2', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix)) 

    # -------------------------------------------------------------------
    # PS-Parameter Settings - (automatic tuning) using RandomSearchCV()
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Algorithm initialisation - PS - 3 
    model = SVC()
    parameters = {'C' : sp_randFloat(),
                  'kernel' : ['linear', 'poly', 'rbf',
                              'sigmoid'],
                  'gamma' : ['auto'],
                  'cache_size' : sp_randInt(100,500),
                  # Add more parameters here for tuning
                 }
    randm = RandomizedSearchCV(estimator=model, 
                        param_distributions = parameters,
                        cv = KFold, n_iter = 10, verbose = 0,
                        n_jobs = 4)
    randm.fit(train_X, train_Y)

    # Results from Random Search
    print("\n========================================================")
    print(" Results from Random Search " )
    print("========================================================")    
    print("\n The best estimator across ALL searched params:\n",
          randm.best_estimator_)
    print("\n The best score across ALL searched params:\n",
          randm.best_score_)
    print("\n The best parameters across ALL searched params:\n",
          randm.best_params_)
    print("\n ========================================================")

    # ----------------------------------------
    # Model setup using random search results
    # ----------------------------------------
    model = randm.best_estimator_
    model = Pipeline([('StandardScaler', StandardScaler(with_mean=False, 
                                                        with_std=False)),
                      #('RobustScaler', RobustScaler()),
                      #('Normilizer', Normalizer()),
                      ('SVC_3', model)])      

    # Cross Validation
    cv_results = cross_val_score(model, train_X, train_Y, 
                                 cv = KFold, scoring='accuracy', 
                                 n_jobs = 4, verbose = 0)
    cv_outcomes.append(cv_results)
    description.append('SVC_3')
    
    # Cross Validation Results
    print("\n%s: " % ('SVM Algorithm: PS-3'))    
    prt_string = "CV Mean Accuracy: %f (Std: %f)"% (
                                    cv_results.mean(), 
                                    cv_results.std())
    print(prt_string)
    
    # Train the Model
    trained_Model = model.fit(train_X, train_Y)

    # Evaluate the skill of the Trained model
    pred_Class          = trained_Model.predict(test_X)
    acc                 = accuracy_score(test_Y, pred_Class)
    classReport         = classification_report(test_Y, pred_Class)
    confMatrix          = confusion_matrix(test_Y, pred_Class) 
    kappa_score         = cohen_kappa_score(test_Y, pred_Class) 

    # collect performance results for further reporting
    model_List.append(('SVC_3', 'SVC Algorithm: PS-3', 
                        trained_Model, acc, kappa_score, 
                        classReport, confMatrix))

    # -------------------------------------------------------------------        
    # Visualise the outcomes / results from Cross Validation
    # -------------------------------------------------------------------    
    fig = pyplot.figure()
    fig.suptitle('Cross Validation Results (Algorithm Comparison)')
    ax = fig.add_subplot(111)
    #pyplot.boxplot(cv_outcomes, vert = True)
    #ax.set_xticklabels(shortDescription)
    pyplot.boxplot(cv_outcomes, vert = False)
    ax.set_yticklabels(description)
    pyplot.show()

    # -------------------------------------------------------------------        
    # Evaluation and Reporting on trained Models
    # -------------------------------------------------------------------        
    print('\nEvaluation and Reporting on trained Models ... ... ... ')
    for shtDes, des, model, accu, kappa, rept, cm in model_List:   
        prt_="\nModel:{M}\nAccuracy:{A}\tKappa:{K}\nReport:\n{R}".format(
                                    M=des, A=round(accu,2), 
                                    K=round(kappa,2), R=rept)
        prt_cm="\nConfusion Matrix:\n{CM}".format(CM=cm)
        print(prt_, prt_cm)
        
        # Save the trained Model
        with open('model_'+shtDes+'.pickle', 'wb') as f:
                pk.dump(model, f)

# ----------------------------------------------------------------------
        # KAPPA (kappa_score) Interpretation:
        # In plain English, 
        #    it measures how much better 
        #    the classier is comparing with guessing 
        #    with the target distribution.
        # Poor agreement        = 0.20 or less
        # Fair agreement        = 0.20 to 0.40
        # Moderate agreement    = 0.40 to 0.60
        # Good agreement        = 0.60 to 0.80
        # Very good agreement   = 0.80 to 1.00
# ----------------------------------------------------------------------
    print("\n\nTrained models are saved on DISK... ... Done ...")       

# ----------------------------------------------------------------------
# Load a (new or existing) dataset to make prediction 
# ----------------------------------------------------------------------     
def loadPredictionDataset():
    engine_str = (
      'mysql+pymysql://{user}:{password}@{server}/{database}'.format(
       user      =  'root',
       password  =  'root888',
       server    =  'localhost',
       database  =  'datasciencerecipes'))
    engine = sa.create_engine(engine_str)
    conn = engine.connect()

    #check whether connection is Successful or not
    #if (conn): print("MySQL Connection is Successful ... ... ...")    
    #else: print("MySQL Connection is not Successful ... ... ...")

    # MySQL Query - New Query is required for Prediction DataSet
    query = '''
    SELECT  
        ClumpThickness, 
        CellSize, CellShape, MarginalAdhesion, 
        EpithelialCellSize, BareNuclei, BlandChromatin, 
        NormalNucleoli, Mitoses
    FROM breastcancerdata;
    '''
    query_result = conn.execute(query)
    dataset =  pd.DataFrame(query_result.fetchall(), 
                            columns =  query_result.keys())
    #print('DataFrame Size',dataset.shape)
    #print('ROW',dataset.shape[0])
    #print('COLUMN',dataset.shape[1])
    conn.close()

    # drop 'SampleCodeNumber' from the training dataset
    #dataset = dataset.drop(['SampleCodeNumber'], axis = 1)
    
    #print('\n')    
    # Get Information on the Dataset
    #print(dataset.info())
    
    # Count Number of Missing Value on Each Column    
    #print('\nCount Number of Missing Value on Each Column: ')        
    #print(dataset.isnull().sum(axis=0))
    # Count Number of Missing Value on Each Row    
    #print('\nCount Number of Missing Value on Each Row: ')        
    #print(dataset.isnull().sum(axis=1))

    # --------------------------------------------------
    # Missing Values treatment of the DataSet (if any)
    # --------------------------------------------------    
    # a) Filling NULL values with Zeros
    #dataset = dataset.fillna(0)
    #print('\nCount Number of Missing Value on Each Column: ')        
    ## Count Number of Missing Value on Each Column
    #print(dataset.isnull().sum(axis=0))
    #print('\nCount Number of Missing Value on Each Row: ')        
    ## Count Number of Missing Value on Each Row
    #print(dataset.isnull().sum(axis=1))

    # b) Filling NULL values according to their dataTypes
    # Group Dataset according to different dataTypes
    gd = dataset.columns.to_series().groupby(dataset.dtypes).groups
    print('\nGroup Columns according to their dataTypes: \n', gd)  
    colNames = dataset.columns.values.tolist()
    
    #dataset[['Class']] = dataset[['Class']].astype('string')
    #dataset[['BareNuclei']] = dataset[['BareNuclei']].astype('int64')
    
    for colName in colNames:
        if dataset[colName].dtypes == 'int64':
           dataset[colName] = dataset[colName].fillna(0)
        if dataset[colName].dtypes == 'float64':
           dataset[colName] = dataset[colName].fillna(0.0) 
        if dataset[colName].dtypes == 'object':
           dataset[colName] = dataset[colName].fillna('Unknown') 
    
    ## Count Number of Missing Value on Each Column
    #print('\nCount Number of Missing Value on Each Column: ')        
    #print(dataset.isnull().sum(axis=0))
    ## Count Number of Missing Value on Each Row    
    #print('\nCount Number of Missing Value on Each Row: ')        
    #print(dataset.isnull().sum(axis=1))

    #dataset.ix[:, 'NumericalClass'] = 0
    #dataset.ix[dataset.Class == 'M', 'NumericalClass'] = 1
    #dataset.ix[dataset.Class == 'R', 'NumericalClass'] = 0
    
    #print('COLUMN',dataset.shape[1])

    return dataset

# ----------------------------------------------------------------------
# How to Rank and Plot features of a trained model
# ----------------------------------------------------------------------
def featureRank_Analysis(model, pred_dataset, cols):
    print()
    print("Feature Importance/Rank Analysis: ")
    X = pred_dataset.ix[:, cols]
    X_cols = X.columns.values
    
    # Without Pipeline
    #features_imp = model.feature_importances_
    
    # With Pipeline
    features_imp = model.steps[1][1].feature_importances_    
    
    indices = np.argsort(features_imp)[::-1]
    df = {}
    for f in range(X.shape[1]):
        print("%d. feature %d %s (%f)" % (f + 1, 
                      indices[f], 
                      X_cols[indices[f]], 
                      features_imp[indices[f]]))
        
        df[f] = [f + 1, indices[f], 
                 X_cols[indices[f]], 
                 features_imp[indices[f]]]
        
    df1 = pd.DataFrame.from_dict(df, orient = 'index')
    df1.columns = ['feature_Rank', 'feature_Index', 
                   'feature_Name', 'feature_importance']
    df1.to_csv("FeatureImportanceRank.csv", index = False)

    # this creates a figure 11 inch wide, 9 inch high
    pyplot.figure(figsize=(11,9)) 
    pyplot.barh(-df1['feature_Rank'], 
                df1['feature_importance'],
                tick_label = df1['feature_Name']
                )
    pyplot.show()    
    
    # this creates a figure 11 inch wide, 9 inch high
    pyplot.figure(figsize=(11,9)) 
    pyplot.bar(df1['feature_Rank'], 
               df1['feature_importance'],
               tick_label = df1['feature_Name']
               )
    pyplot.xticks(rotation=90)
    pyplot.show()

    # this creates a figure 11 inch wide, 9 inch high
    pyplot.figure(figsize=(11,9)) 
    pyplot.barh(df1['feature_Rank'], 
                df1['feature_importance'],
                tick_label = df1['feature_Name']
                )
    #plt.savefig('Featurefig.pdf', format='pdf')
    pyplot.show()   

# ----------------------------------------------------------------------
# Load the trained model and make prediction
# ----------------------------------------------------------------------
def loadTrainedModelForPrediction(dataset):
    f = open('model_MLP_1.pickle', 'rb')
    model = pk.load(f); f.close();
    
    # Separate out Numerical & Categorical Variables
    colNumeric = []; colCategory = [];
    colNames = dataset.columns.values.tolist()

    for colName in colNames:
        if dataset[colName].dtypes == 'int64':
           colNumeric.append(colName)
        if dataset[colName].dtypes == 'float64':
           colNumeric.append(colName)
        if dataset[colName].dtypes == 'object':
           colCategory.append(colName)

    print()           
    print('Number of Categocial & Numerical Features: ', 
                                            len(colNames)) 
    print('Number of Categocial Features: ', len(colCategory))
    print('Number of Numerical Features: ', len(colNumeric))
   
    
    # --------------------------
    # Only Numerical Features
    # --------------------------    
    cols_X = colNumeric; #print(cols_X)

    # ---------------------------------------    
    # ALL Numerical & Categorical Features with Encoding
    # ---------------------------------------    
    #cols_X = colNames;  #print(cols_X)
    #p_dataset = dummyEncode(dataset.loc[:, cols_X])        

     
    pred_Value = model.predict(dataset[cols_X])
    dataset.loc[:, 'PredictionResult'] = pred_Value
    
    # Feature Ranking Analysis 
    #featureRank_Analysis(model, dataset, cols_X)
    
    return dataset

# ----------------------------------------------------------------------
# Finalise the results and update the audiance
# ----------------------------------------------------------------------    
def finaliseResult(result):
    # Save Result in a CSV file
    result.to_csv('finalResult.csv', index = False)
    print("\n\nSave Result in a CSV file ... ... Done ...")    
    
    # Save Result in a MySQl Table
    engine_str = (
      'mysql+pymysql://{user}:{password}@{server}/{database}'.format(
       user      =  'root',
       password  =  'root888',
       server    =  'localhost',
       database  =  'datasciencerecipes'))
    engine = sa.create_engine(engine_str)
    conn = engine.connect()
    
    #check whether connection is Successful or not
    #if (conn): print("MySQL Connection is Successful ... ... ...")    
    #else: print("MySQL Connection is not Successful ... ... ...")

    result.to_sql(name='breastcancerresult', con=engine, 
                  schema='datasciencerecipes', 
                  if_exists = 'replace', chunksize = 1000, 
                  index=False)
    print("Save Result in a MySQl Table ... ... Done ...")        
    conn.close()

# ----------------------------------------------------------------------
# End-2-End Applied Machine Learning Recipe for Beginners or Developers  
# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    # -----------------------------------------------------------------
    start_time = time.time()
    # -----------------------------------------------------------------
    filename = 'BreastCancerWisconsin.data.csv'

    # -----------------------------------------------------------------
    dataset = loadFrCSVFile(filename)
    # -----------------------------------------------------------------
    import2MySQL(dataset)
    # -----------------------------------------------------------------
    dataset = loadDataSetFrMySQLTable()
    # -----------------------------------------------------------------
    summariseDataset(dataset)
    # -----------------------------------------------------------------
    visualiseDataset(dataset)
    # -----------------------------------------------------------------
    train_X, test_X, train_Y, test_Y = preProcessingData(dataset)
    # -----------------------------------------------------------------
    evaluateAlgorithm(train_X, test_X, train_Y, test_Y, dataset)
    # -----------------------------------------------------------------
    pred_Dataset = loadPredictionDataset()
    # -----------------------------------------------------------------
    result = loadTrainedModelForPrediction(pred_Dataset)
    # -----------------------------------------------------------------
    finaliseResult(result)
    # -----------------------------------------------------------------
    print()
    print("Execution Time %s seconds: " % (time.time() - start_time))
    # -----------------------------------------------------------------
