
# coding: utf-8
# change
# In[1]:


get_ipython().magic('matplotlib inline')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from collections import Counter


# In[3]:


from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[4]:


file = r"/Users/mars/OneDrive/Data Mining/Project/CS235ProjectCode/batch1.dat" ##training data set


# In[5]:


def data_parse(line):
    lp = line.split(' ')
    gas = int(lp[0].split(';')[0])
    ppm = float(lp[0].split(';')[1])
    dataSet = list()
    for i in range(16):
        dataItem = list()
        for j in range(7):
            dataItem.extend([float(lp[i*8+j+2].split(':')[1])])
        dataSet.append(dataItem)
    return gas,ppm,dataSet


# In[6]:


#function to read and format data

def read_data(file, table, sensors):
    fileData = open(file, 'r')
    while 1:
        line = fileData.readline()
        if not line:
            break
        #print(line)
        gas = int(line[0])
       
        data = data_parse(line)
        for i in range(len(sensors)):
            currentData = table.loc[gases.index[gas-1],sensors[i]]
            if currentData.size == 0:
                table.loc[gases.index[gas-1],sensors[i]] = np.array([data[2][i]])
            else:
                 table.loc[gases.index[gas-1],sensors[i]] = np.append(currentData,[data[2][i]],axis = 0)
    fileData.close()


# In[7]:


# repalce the NaN element with the empty list to facillate the following process
def init_dTable(table):
    for index in table.index:
        for column in table.columns:
            table.loc[index, column] = np.array([[]])


# In[8]:


# output the table size
def data_size(data_table):
    size_table = empty_table.copy()
    for column in size_table.columns:
        size_table[column] = data_table[column].apply(np.shape)
    return size_table


# In[9]:


# statistic num of curves
def n_of_curves(data_table):
    n = 0
    for index in data_table.index:
        n += sum(map(len,(data_table.loc[index])))
    print("The dataset has all together {:d} curves".format(n))
    return n


# In[10]:


# plot data table
def data_plot(data_table):
    m,n = np.shape(data_table)
    fig = plt.figure(figsize =(20, 2*m))
    for i in range(m):
        for j in range(n):
            fig.add_subplot(m,n,(i*n+j+1))
            plt.plot((data_table.iloc[i,j]).T)
    # Add sensor names
    for s in range(n):
        fig.add_subplot(m,n,s+1).set_title(sensors[s]);
    # Add gas names
    for g in range(m):
        fig.add_subplot(m,n,g*n+1)
        plt.ylabel(gases.index[g], fontsize=10)


# In[11]:


# using preprocess package to scale the curves
def data_scaling(data_table):
    data_table_scaled = empty_table.copy()
    for index in data_table.index:
        for column in data_table.columns:
            data_table_scaled.loc[index,column] = preprocessing.scale(data_table.loc[index,column], axis =1)
    return data_table_scaled


# In[12]:


# deleting curves from data_table, the index of curve is in detetion_table
def data_cleaning(data_table, deletion_table):
    data_table_cleaned = empty_table.copy()
    for index in data_table.index:
        for column in data_table.columns:
            data_table_cleaned.loc[index,column] = np.delete(data_table.loc[index,column], deletion_table
                                                             .loc[index,column], axis =0)
    return data_table_cleaned


# In[13]:


# trainin the SVM model
def multi_class_clf(X_list, n_fold, n_shuffle):
    # Create feature matrix
    X = np.concatenate(X_list, axis=0)
    # Create label vector
    y = ()
    for i in range(len(X_list)):
        y = np.concatenate((y, (np.ones(len(X_list[i]), int) * i)), axis=0)

    # Create classifier object
    clf = OneVsRestClassifier(LinearSVC())
    all_scores = []
    for _ in range(n_shuffle):
        skf = StratifiedKFold(n_fold, shuffle=True)
        #skf = skf.get_n_splits(X_list,y)
        # kf = KFold(len(X),2,shuffle = True)
        results = []
        for train_index, test_index in skf.split(X,y):
            #print("TRAIN",train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            results.append(clf.fit(X_train, y_train).score(X_test, y_test))  # results[] contains n_fold scores
        all_scores.append(np.mean(results))  # all_scores[] contains n_shuffle scores
    s = round(np.mean(all_scores), 4)
    # print "The average {:d}-fold cross-validation accuracy over {:d} shuffles is: {:.2f}%".format(n_fold, n_shuffle, s*100)
    return s, clf


# In[14]:


# funtion to check whether the curve is beyond mean +- standard deviation
def eli_err(x_list):
    err_list = list()
    for i in range(len(x_list)):
        if (x_list[i] < x_list.mean()-x_list.std()) or (x_list[i] > x_list.mean()+x_list.std()):
            err_list.append(i)
    return err_list


# In[15]:


# generate deleting table
def gen_delete_table(data_table):
    clean_table = empty_table.copy()
    for index in data_table.index:
        for column in data_table.columns:
            clean_data = list()
            for i in range(7):
                data = data_table.loc[index, column].T[i]
                clean_data.extend(eli_err(data))
            clean_table.loc[index,column] = list(set(clean_data))
    return clean_table


# In[16]:


# index and names used to generte pandas dataframe
gases = pd.Series(('ethl', 'ethe', 'ammo', 'atal', 'acet', 'tolue'),index = ["ethanol","ethylene","ammonia","acetaldehyde","acetone","toluene"])

#sensors = ["1", "2", "3", "4", "5", "6", "7", "8"]
sensors = ["1", "2", "3", "4", "5", "6", "7", "8","9","10","11","12","13","14","15","16"]


# In[17]:


empty_table = pd.DataFrame(index = gases.index, columns = sensors)

name_table = empty_table.copy()


# In[18]:


for index in gases.index:
    for column in sensors:
        name_table.loc[index, column] = "%s_%s"%(gases[index], column)


# In[19]:


# import data from training set as data_table_raw
data_table_raw = empty_table.copy()
init_dTable(data_table_raw)
read_data(file, data_table_raw, sensors)
data_size(data_table_raw)


# In[20]:


# plot the data_table_raw
data_plot(data_table_raw)


# In[21]:


data_table_scaled = data_scaling(data_table_raw)


# In[22]:


# generate cleaned data_table
data_table_cleaned = data_cleaning(data_table_scaled,gen_delete_table(data_table_scaled))
#data_plot(data_table_cleaned)


# In[23]:


data_size(data_table_cleaned)


# In[24]:


scoreMatrix = pd.DataFrame(index = ['scaled','cleaned'], columns = sensors)


# In[25]:


# generate SVM model and using K-fold to valid the model
clfs = scoreMatrix.copy()
for index in scoreMatrix.index:
    for column in scoreMatrix.columns:
        X_list = eval('data_table_'+index+'[column]'+'.values')
        scoreMatrix.loc[index,column], clfs.loc[index,column] = multi_class_clf(X_list, 5,20)
scoreMatrix.plot.bar()
scoreMatrix


# In[ ]:





# In[38]:


# generate concentration table
def generate_con_table(filename,norm):
    consList = list()
    for i in range(len(gases)):
        exec(gases[i]+"_table = pd.DataFrame(columns = sensors+['cons'])")
        
    fileData = open(file,'r')
    i = 0
    while 1:
        line = fileData.readline()
        if not line:
            break
        gas = int(line.split(' ')[0].split(';')[0])
        ppm = float(line.split(' ')[0].split(';')[1])
        
        dataItem = list()
        for j in range(16):
            dataItem.extend([float(line.split(' ')[j*8+1+norm].split(':')[1])])
            
        dataItem.append(ppm)
        itemName = gases.iloc[gas-1]+ "_table.loc[" + str(i)+ "]"
        exec(itemName + "= dataItem")       
        
        i = i+1
        
    for i in range(len(gases)):
        consList.append(eval(gases[i]+"_table"))
                        
    return consList


# In[66]:


# generate the regression table
def tables_for_reg(gas_list, concList):

    #feature_table = pd.DataFrame(index = gas_list, columns = sensor_list)
    #target_table = pd.DataFrame(index = gas_list, columns = sensor_list)
   
    r2score_table = pd.DataFrame(index = gas_list, columns = ["coefficient", "intersect", "score"])
    linreg = LinearRegression()
    for g in range(len(gas_list)):
        #for s in len(sensors):
            
        x = concList[g].ix[:,:-1]
        y = concList[g].ix[:,-1]            
            
        linreg.fit(x,y)
            
            #reg_table.iloc[g,s] = (linreg.coef_,linreg.intercept_)
        r2score_table.loc[gases[g]]=[linreg.coef_, linreg.intercept_, round(r2_score(y,linreg.predict(x)),3)]
    return r2score_table

concList = generate_con_table(file,1)
reg_table = tables_for_reg(gases, concList)

reg_table


# In[76]:





# In[31]:


# predict the gas class of the sensing data in 'line'
def cls_pred(clfs,sel,line):
    result = list()
    for s in range(16):
        x = list()
        for i in range(7):
            x.append(float(line.split(' ')[s*8+i+2].split(':')[1]))
         
        #print(np.array([x]))
        x_scaled = preprocessing.scale(np.array([x]), axis =1)
        result.append(int(clfs.iloc[(sel,s)].predict(x_scaled)))
    return result


# In[32]:


# not used
"""
def predict_file(file, clfs, sel):
    tf = open(file)
    while 1:
        line = tf.readline()
        if not line:
            break
        gas = list([int(line.split(' ')[0].split(';')[0])-1])
        
        gas.extend(cls_pred(clfs, sel, line))
        print(gas)
"""


# In[33]:


# predict the gas type using SVM model

def predict(line, clfs, sel, clasNum):
    gas = list([int(line.split(' ')[0].split(';')[0])-1])
    ppm = list([line.split(' ')[0].split(';')[1]])
    pred_List = cls_pred(clfs, sel, line)
    filt_List = list()
    for num in clasNum:
        filt_List.append(pred_List[num])
    aa = Counter(filt_List)
    return gas[0], aa.most_common(1)[0][0], ppm


# In[47]:


# predict the gas concentration use regression

def pred_cons(gas, cons_table, line, norm):
    dataSen = list()
    for j in range(16):
        dataSen.extend([float(line.split(' ')[j*8+1+norm].split(':')[1])])
    
        
    List3 = np.multiply(reg_table.iloc[(gas,0)],np.array(dataSen))

    return List3.sum()+cons_table.iloc[(gas,1)]


# In[88]:


# import the test set, predict gas type and concentration
testFile = r'/home/zhaoyuan/testSet_2.txt'
tf = open(testFile)
hit = 0
sumNum = 0
while 1:
    line = tf.readline()
    if not line:
        break
    
    result = predict(line, clfs, 1, [1,2,9,10])
    gas = result[0]
    predGas = result[1]
    ppm = result[2]
    
    #print(predGas, gas, pred_cons(predGas, reg_table, line, 1), ppm)
    
    if(gas == predGas):
        hit = hit+1
    sumNum = sumNum+1
       
print(hit, sumNum, hit/sumNum)

