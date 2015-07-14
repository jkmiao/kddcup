#!/usr/bin/env python
# coding=utf-8

import pickle
import pandas as pd
import numpy as np


source_dict={'server':0,'browser':1}
event_dict = {"problem":5,"video":3,"access":1,"wiki":4,"discussion":6,"navigate":2,"page_close":0}


def gen_time_dict():
    rng = pd.date_range('2013-10-27','2014-08-01')
    time_dict = pd.Series(np.arange(len(rng)),index=rng)
    fw = open('data/time_dict.csv','w')
    pickle.dump(time_dict,fw)
    return time_dict


def gen_courseid_dict():
    df = pd.read_csv('data/date.csv',usecols=[0])
    course_map = pd.factorize(df.course_id)[1]
    course_dict = dict(zip(course_map,range(len(course_map))))
    fw = open('data/course_idTrain2.csv','w')
    pickle.dump(course_dict,fw)
    print "course_dict done"
    return course_dict


def gen_object_dict():
    df = pd.read_csv('data/log_train.csv',usecols=[4])
    obj_map = pd.factorize(df.object)[1]
    obj_dict = dict(zip(obj_map,range(len(obj_map))))
    
    df2 = pd.read_csv('data/test/log_test.csv',usecols=[4])
    obj_map2 = pd.factorize(df2.object)[1]
    diff = [w for w in obj_map2 if w not in obj_map]
    obj_dict2 =dict(zip(diff,np.arange(len(obj_map),len(obj_map)+len(diff))))
    
    obj_dict.update(obj_dict2)
    fw = open('data/object_pkl.csv','w')
    pickle.dump(obj_dict,fw)
    print "obj_dict done.."
    return obj_dict


def time_map(x):
    x = x[:10]
    return time_dict[x]


def obj_map(x):
    return obj_dict[x]


def course_map(x):
    return course_dict[x]

time_dict = gen_time_dict()
course_dict= gen_courseid_dict()
obj_dict= gen_object_dict()



def log_trainData():
    print "read log_train.csv "
    df1 = pd.read_csv('data/log_train.csv',converters={1:time_map,4:obj_map})
    print df1.head()
    
    df1.source = df1.source.map(lambda x:source_dict[x])
    df1.event = df1.event.map(lambda x:event_dict[x])
    print df1.head()
    print df1.tail()
    df1.to_csv('data/log_trainData.csv',index=False)
    

def course_Data():
    df2 = pd.read_csv('data/enrollment_train.csv',usecols=[0,2],converters={2:course_map})
    df3 = pd.read_csv('data/date.csv',converters={0:course_map,1:time_map,2:time_map})
    df4 = pd.merge(df2,df3,on='course_id',how='outer')
    df4 = df4.sort_index(by='enrollment_id')
    print df4.tail(10)
    df4.to_csv("data/course_Trainpkl.csv",index=False)

    df1 = pd.read_csv('data/test/enrollment_test.csv',usecols=[0,2],converters={2:course_map})
    df4 = pd.merge(df1,df3)
    df4 = df4.sort_index(by='enrollment_id')
    print df4.tail(10)
    df4.to_csv("data/test/course_Testpkl.csv",index=False)



def log_testData():
    print "read log_test.csv "
    df1 = pd.read_csv('data/test/log_test.csv',converters={1:time_map,4:obj_map})
    print df1.tail(10)
    df1.source = df1.source.map(lambda x:source_dict[x])
    df1.event = df1.event.map(lambda x:event_dict[x])
    print df1.tail(10)
    df1.to_csv('data/test/log_testData.csv',index=False)

log_trainData()
log_testData()
course_Data()
