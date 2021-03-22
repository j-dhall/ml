#!/usr/bin/env python
# coding: utf-8

# In[1]:


#usage
#config = Config ()
#config.set_dataset_id ("boston-housing")
#config.get_train_df ().head(5)

import os
import json
import pandas as pd
class Config:
    def __init__ (self, config_file = 'config.json'):
        with open (config_file, 'r') as f:
            self.config = json.load (f)

        self.dataset_id = None
        self.train_f = None
        self.test_f = None
    
    def set_dataset_id (self, id):
        self.dataset_id = id
        datasets_dir = self.config ["data"]["datasets-dir"]
        dataset_dir = self.config ["data"]["datasets"][id]["dir"]
        train_f = self.config ["data"]["datasets"][id]["files"]["train"]
        test_f = self.config ["data"]["datasets"][id]["files"]["test"]
        self.train_f = os.path.join (datasets_dir, dataset_dir, train_f)
        self.test_f = os.path.join (datasets_dir, dataset_dir, test_f)

    def is_dataset_id_valid (self):
        return self.dataset_id != None
    
    def get_train_f (self):
        if not self.is_dataset_id_valid ():
            return None
        return self.train_f
    
    def get_test_f (self):
        if not self.is_dataset_id_valid ():
            return None
        return self.test_f
    
    def get_train_df (self):
        return pd.read_csv (self.train_f)
    
    def get_test_df (self):
        return pd.read_csv (self.test_f)


# In[ ]:




