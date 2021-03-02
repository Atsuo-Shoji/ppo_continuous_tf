# -*- coding: utf-8 -*-
import numpy as np
import pickle


###Utility Functions###


def read_pickle_file(file_path):
    #指定されたパスのpickleファイルを読み込む。
    
    with open(file_path, "rb") as fo:
        obj = pickle.load(fo)
        
    return obj

        
def save_pickle_file(obj, file_path):
    #指定されたオブジェクトを指定されたパスのpickleファイルとして書き込む。
    
    with open(file_path, 'wb') as fo:
        pickle.dump(obj , fo) 
        
        
def timedelta_HMS_string(td):
    
    hours = td.seconds//3600
    remainder_secs = td.seconds%3600
    minutes = remainder_secs//60
    seconds = remainder_secs%60
    
    HMS_string = str(hours) + " hours " + str(minutes) + " minutes " + str(seconds) + " seconds"
    
    return HMS_string


def standardize(obj_data, with_mean=True):
    #標準化関数
    
    epsilon = 1e-6
    
    std = np.std(obj_data)
    
    if with_mean==True:
        avg = np.average(obj_data)
        ret = (obj_data - avg) / (std + epsilon)
    else:
        ret = obj_data / (std + epsilon)
    
    return ret


###Utility Functions　終わり###
   
    
###class Calculater_Statistics###


class Calculater_Statistics:
        
    #ある分布の平均と分散を逐次計算
    #対象統計データ量が多くなる場合に使用（多くなければ本クラスを使用せず、例えば標準偏差ならnp.stdでよい）
    #よって、対象統計データそのものを保持せず、その平均と分散のみを保持する。
    #追加データを加味した平均と分散を追加都度計算し直し、それらを返す。
        
    id_instance = 0
        
    def __init__(self, id, shape=None, dtype=np.float32, description=""):
            
        self._mean_curr = np.zeros(shape, dtype) #現時点での対象統計データの平均
        self._var_curr = np.zeros(shape, dtype) #現時点での対象統計データの分散
        self._count_curr = 0 #現時点での対象統計データの件数
            
        self._shape = shape
        self._dtype = dtype
            
        self._id = id #インスタンスが複数作成される場合の区別に使用          
        self._description = description #このインスタンスの使用目的に使用　「to calculate the stdev of the accumulated rewards」など
        
    @classmethod
    def createInstance(cls, shape=None, dtype=np.float32, description=""):
            
        id_ins = cls.id_instance
        ins = cls(id_ins, shape, dtype, description)  
            
        cls.id_instance += 1
            
        return ins
        
    def update_mean_var(self, X_added):
            
        #X_added：追加するデータ　　shapeは(追加データ数, shape)
        #ただし、shapeがNoneや()の場合、shapeは(追加データ数, )
        #　使用される場面は、train()中あるエポックでのtrajectory収集後の蓄積rewardの標準偏差算出
        #　X_addedは、そのエポックでの全stepのrewardsのndarray　shapeは(このエポックでのstep数, 1)
            
        mean_added = np.mean(X_added, axis=0)
        var_added = np.var(X_added, axis=0)
        count_added = X_added.shape[0]
            
        mean_delta = mean_added - self._mean_curr
        var_added_multiplied_count = var_added * count_added
        var_curr_multiplied_count = self._var_curr * self._count_curr
            
        count_new = self._count_curr + count_added
        mean_new = self._mean_curr + ( mean_delta * count_added / count_new )            
        var_new = (  var_curr_multiplied_count + var_added_multiplied_count \
                    + ( np.square(mean_delta) * self._count_curr * count_added / count_new )  ) \
                    / count_new
            
        self._mean_curr = mean_new
        self._var_curr = var_new
        self._count_curr = count_new
            
        return mean_new, var_new, np.sqrt(var_new)                         
            
        
    @property
    def id(self):
        return self._id
        
    @property
    def shape(self):
        return self._shape
        
    @property
    def dtype(self):
        return self._dtype
        
    @property
    def description(self):
        return self._description
        
    @property
    def curr_mean(self):
        return self._mean_curr
        
    @property
    def curr_var(self):
        return self._curr_var
        
    @property
    def curr_stdev(self):
        return np.sqrt(self._curr_var)


###class Calculater_Statistics　終わり###
