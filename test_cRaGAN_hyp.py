# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:49:41 2020

@author: baekd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from utils.misc import read_json
from utils.cRaGAN_hyp import cRaGAN
import argparse

"""
    main entry for the script
"""
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',type=str,required=True)
    parser.add_argument('-nch',type=int,required=True)
    parser.add_argument('-blk',type=int,required=True)
    args = parser.parse_args()
    json_dir = args.opt

    model = cRaGAN(json_dir,cuda=True,nch=args.nch,num_blks_mid=args.blk)
    model.test_final()

            
            
    
    


    







































