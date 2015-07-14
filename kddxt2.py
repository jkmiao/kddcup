#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import pickle

from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import scale

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score



def norm( x ):
    if x<0.000001:
        x=0
    elif x>0.96:
        x=1
    return x

def last_time(x):
    return x.max()-x.min()

def loadTrainData():
    df1 = pd.read_csv('data/log_trainData.csv')
    print df1.head()
    print df1.tail()
    df2 = pd.read_csv('data/truth_train.csv',header=None,usecols=[1],names=["drop"])
    df3 = pd.read_csv('data/course_Trainpkl.csv',usecols=[1,2,3])
    
    
    gp = df1.groupby("enrollment_id")

    data = df1.pivot_table("source",rows='enrollment_id',cols="event",aggfunc='count',fill_value=0)
    
    eventdf = gp.event.describe().unstack()

    timedf = gp.time.describe().unstack()
    timedf.drop('count',axis=1)

    sourcedf = gp.source.describe().unstack()
    sourcedf.drop(['count','min','max'],axis=1)

    objectdf = gp.object.describe().unstack()
    objectdf.drop(['count'],axis=1)

    data = pd.concat([data,eventdf],axis=1)
    data = pd.concat([data,timedf],axis=1)
    data = pd.concat([data,sourcedf],axis=1)
    data = pd.concat([data,objectdf],axis=1)
    
    data['dtime'] = gp.time.apply(last_time)
    data["course_id"] = df3["course_id"].values
    data["from"] = df3["from"].values
    data["to"] = df3["to"].values
    
    # X = MinMaxScaler().fit_transform(X)
    print "origin data: "
    print data.tail() 
    data = data.fillna(0)
    data.to_csv('data/trainData.csv',index=False)
    X = data.values 
    X = scale(X)
    
    df2.to_csv('data/trainLabel.csv',index=False)
    y = np.ravel(df2['drop'])   
    print "y: ",y[:5]
    return X,y

def loadTestData():
    df1 = pd.read_csv('data/test/log_testData.csv')
    print df1.head()
    print df1.tail()
    df3 = pd.read_csv('data/test/course_Testpkl.csv',usecols=[1,2,3])

    
    gp = df1.groupby("enrollment_id")

    data = df1.pivot_table("source",rows='enrollment_id',cols="event",aggfunc='count',fill_value=0)
    
    eventdf = gp.event.describe().unstack()

    timedf = gp.time.describe().unstack()
    timedf.drop('count',axis=1)

    sourcedf = gp.source.describe().unstack()
    sourcedf.drop(['count','min','max'],axis=1)

    objectdf = gp.object.describe().unstack()
    objectdf.drop(['count'],axis=1)

    data = pd.concat([data,eventdf],axis=1)
    data = pd.concat([data,timedf],axis=1)
    data = pd.concat([data,sourcedf],axis=1)
    data = pd.concat([data,objectdf],axis=1)

    data['dtime'] = gp.time.apply(last_time)
    data["course_id"] = df3["course_id"].values
    data["from"] = df3["from"].values
    data["to"] = df3["to"].values
    
    # data["cnt"]=gp.size()
    # data["eventstd"] = gp.event.std()
    # data['eventmean'] = gp.event.mean()
    # data['eventmdeian'] = gp.event.median()
    # data['equantile0.25'] = gp.event.quantile(0.25)
    # data['equantile0.75'] = gp.event.quantile(0.75)
    # data['equantilemad'] = gp.event.mad()
    
    print "test data: "
    print data.tail(10)
    data = data.fillna(0)
    data.to_csv('data/test/testData.csv',index=False)
    
    test = data.values
    # test = MinMaxScaler().fit_transform(test)
    test = scale(test)
    return test


def svc_clf(x_train,x_test,y_train,y_test,test):
    clf = svm.SVC(kernel='linear',probability=True,random_state=42)
    clf.fit(x_train,y_train)
    y_pred= clf.predict_proba(x_test)[:,1]
    scores = roc_auc_score(y_test,y_pred)
    print "svm scores:...",scores
    pred = clf.predict_proba(test)[:,1]
    saveResult(pred,'data/test/svc_res.csv')


def lr_clf(x_train,x_test,y_train,y_test,test):
    clf = linear_model.LogisticRegression()
    clf.fit(x_train,y_train)
    y_pred = clf.predict_proba(x_test)[:,1]
    scores= roc_auc_score(y_test,y_pred)
    print "lr_clf scores: ",scores
    
    y_pred = map(norm,y_pred)
    score2 = roc_auc_score(y_test,y_pred)
    print "after nomailzied score ... ",score2
    
    pred = clf.predict_proba(test)[:,1]
    saveResult(pred,'data/test/lr_res.csv')

def rf_clf(x_train,x_test,y_train,y_test,test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train,x_train)
    y_pred = clf.predict_proba(x_test)[:,1]
    scores = roc_auc_score(y_test,y_pred)
    pred = clf.predict(test)[:,1]
    print "rf_scores: ",scores
    saveResult(pred,'./data/test/rf_res.csv')


def  gbdt_clf(x_train,x_test,y_train,y_test,test):
    clf = GradientBoostingClassifier(n_estimators=500)
    clf.fit(x_train,y_train)
    y_pred=clf.predict_proba(x_test)[:,1]
    scores = roc_auc_score(y_test,y_pred)
    pred = clf.predict_proba(test)[:,1]
    print "gbdt_clf scores: ",scores
    saveResult(pred,'data/test/gbdt_clf'+str(scores)+'.csv')


def saveResult(pred,fileName):
    df = pd.read_csv('data/test/enrollment_test.csv',usecols=[0])
    df['drop'] = pred
    print df.head()
    df.to_csv(fileName,index=False,header=False)

def em_res():
    df = pd.read_csv("data/test/lr_res.csv",header=None,names=["id","drop"])
    df1 = pd.read_csv("data/test/gbdt_clf0.883858223551.csv",header=None,usecols=[1],names=["drop1"])
    df2 = pd.read_csv("data/test/final_res2.csv",header=None,usecols=[1],names=["drop2"])
    df["drop"]  =df["drop"]*0.2+ df1["drop1"]*0.5+df2["drop2"]*0.3
    df["drop"] = df["drop"]
    df.to_csv("data/test/final_res2.csv",index=None,header=None)


def loadPickleTrainData():
    df1 = pd.read_csv('data/trainData.csv')
    print df1.head()
    print df1[5:6]
    X = df1.values
    # X = scale(X)
    df2 = pd.read_csv('data/trainLabel.csv')
    y = np.ravel(df2.values)
    return X,y


def loadPickleTestData():
    df1 = pd.read_csv('data/test/testData.csv')
    test = df1.values
    # test = scale(test)
    return test


def dropPredict():
    em_res()
    print "loading train data..."
    X,y = loadPickleTrainData()
    # X,y = loadTrainData()

    print "loading test data... "
    test = loadPickleTestData()
    # test = loadTestData()

    print "\nmodeling lr..."
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=18)
    lr_clf(x_train,x_test,y_train,y_test,test)
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.32,random_state=125)

    print "\nmodeling gbdt..."
    gbdt_clf(x_train,x_test,y_train,y_test,test)

    print "\nmodeling svm..."
    # svc_clf(x_train,x_test,y_train,y_test,test)

    print "done"

if __name__ =="__main__":
    print "start>>>"
    dropPredict()
