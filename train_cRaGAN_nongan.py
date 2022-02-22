# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:49:41 2020

@author: baekd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from utils.misc import read_json
from utils.nongan import nongan
import argparse

"""
    main entry for the script
"""
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',type=str,required=True)
    args = parser.parse_args()
    json_dir = args.opt

    model = nongan(json_dir,cuda=True)
    model.train()
            
            
    
    


    







































