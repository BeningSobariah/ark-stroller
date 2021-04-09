# -*- coding: utf-8 -*-
# @Author: Muhammad Alfin N
# @Date:   2021-03-30 20:13:14
# @Last Modified by:   Muhammad Alfin N

# @Last Modified time: 2021-04-07 18:38:41


import numpy as np
import imageio
import random
import torch
import PIL
import os
import cv2
import time
import pyrebase

from models import get_instrumented_model
from decomposition import get_or_compute
from ipywidgets import fixed,Layout
from skimage import img_as_ubyte
from tqdm import tqdm,trange
from config import Config
from PIL import Image

torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

class Model:
    def __init__(self, name, num_components=60):
        
        if name == 'stroller':
            model_class = 'stroller_193'
        if name == 'pushchair':
            model_class = 'pushchair_572'
        if name == 'childseat':
            model_class = 'childseat_719'
            
        self.name = name
        self.config = Config(
                  model='StyleGAN2',
                  layer='style',
                  output_class=model_class,
                  components=num_components,
                  use_w=True,
                  batch_size=5000
                )
        self.model = self.init_model()
        self.storage = self.init_storage()

    def init_storage(self):
        config = {
            "apiKey": "AIzaSyCP9sj_xIogRC_5EowwMwQIh9MEvLlCqrk",
            "authDomain": "niomata-745ae.firebaseapp.com",
            "projectId": "niomata-745ae",
            "storageBucket": "niomata-745ae.appspot.com",
            "messagingSenderId": "933534202946",
            "appId": "1:933534202946:web:8c1d2b2b94b772533a81db",
            "measurementId": "G-MZCLX7LM9G",
            "databaseURL":"https://niomata-745ae-default-rtdb.firebaseio.com/"
        }

        firebase = pyrebase.initialize_app(config)
        return firebase.storage()

    def init_model(self):
        inst = get_instrumented_model(self.config.model, 
                                        self.config.output_class,
                                        self.config.layer, torch.device('cuda'), 
                                        use_w=self.config.use_w)
        model = inst.model
        return model
    
    def normalize_z(self,z):
        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        if self.name == 'stroller':
            good_seed = [2,3,5,6,7,8,12,13,15,19,20,22,30,39,41,42,43,51,57,63,68,72,91,99,102,144,155,158,167,178,187,239,240,243,297,298,322,323,333,334,335,344,370,373,374,376,384,423,425,436,445,447,472,475,484,499,500,527,576,581,582,595,631,671,689,690,698,708,838,895]
        if self.name == 'pushchair':
            good_seed = [2,3,4,5,7,8,9,10,11,12,13,20,21,25,31,50,59,62,63,64,107,108,120,129,134,155,191,217,224,229,230,232,242,243,244,247,250,291,294,326,341,366,369,370,373,385,391,392,393,398,417,425,440,451,459,462,472,494,501,515,522,523,534,525,545,553,559]
        if self.name == 'childseat':
            good_seed = [2,3,27,38,45,48,49,68,78,82,86,90,91,96,110,118,149,154,155,158,159,160,162,167,201,202,206,290,294,295,296,297,302,309,350,351,380,412,437,449,450,451,452,468,500,503,508,519,520,560,561,565,641,643,684,687,715,777,778,813,878,885,889,915,926,927,955,937,972,975,988,993,994]

        base = self.model.sample_latent(1, seed=random.choice(good_seed)).cpu().numpy()
        z = base + (z - base)*0.5

        return z
    
    def generate_from_z(self,z,truncation=0.5,resize=(500,500),normalize=True):
        self.model.truncation = truncation
        z = self.normalize_z(z) if normalize else z
        img = self.model.sample_np(z)
        img = Image.fromarray((img * 255).astype(np.uint8)).resize(resize,Image.LANCZOS)

        return img
    
    def generate_image(self,seed):
        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        z = self.model.sample_latent(1, seed=seed).cpu().numpy()
        return self.generate_from_z(z)
    
    def generate_z(self,seed):
        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        return self.model.sample_latent(1, seed=seed).cpu().numpy()

    def generate_transversal(self,output_dir,combinations,num):
        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        seeds = []

        for i in num:
            z = self.model.sample_latent(1, seed=i).cpu().numpy()
            z = self.normalize_z(z)
            seeds.append(z)

        for combi in combinations:
            seed_1 = seeds[combi[0]-1]
            seed_2 = seeds[combi[1]-1]
            step = 5

            imgs = self.transverse_image(z_1=seed_1,
                                            z_2=seed_2,
                                            step=step)

            seed_1_path = os.path.join(output_dir,self.name,'{}_{}.npy'.format(self.name,combi[0])).replace(os.sep,'/')
            seed_2_path = os.path.join(output_dir,self.name,'{}_{}.npy'.format(self.name,combi[1])).replace(os.sep,'/')

            np.save(os.path.splitext(seed_1_path)[0],seed_1)
            # self.storage.child(seed_1_path).put(seed_1_path)

            np.save(os.path.splitext(seed_2_path)[0],seed_2)
            # self.storage.child(seed_2_path).put(seed_2_path)
            

            for step,img in enumerate(imgs):
                if step == 0:
                    name = '{}_{}.jpg'.format(self.name,combi[0],step)
                elif step == (len(imgs)-1):
                    name = '{}_{}.jpg'.format(self.name,combi[1],step)
                else:
                    name = '{}_{}_to_{}_{}.jpg'.format(self.name,combi[0],combi[1],step)

                img_path = os.path.join(output_dir,self.name,name).replace(os.sep,'/')
                img.save(img_path)
                self.storage.child(img_path).put(img_path)

    def transverse_image(self,z_1,z_2,step=5):
        zs = []
        
        for i in range(step):
            z = z_1 + 1/step*i*(z_2-z_1)
            zs.append(z)
            
        return [self.generate_from_z(x,normalize=False) for x in zs]

    def generate_style(self,z,data,output_dir):

        id_style    = data['name']
        step        = int(data['step'])
        scale       = int(data['scale'])
        ls          = int(data['layer_start'])
        le          = int(data['layer_end'])
        rule        = np.load(data['path'])
        truncation  = float(data['truncation'])

        index = 0
        output_dir = os.path.join(output_dir,self.name,id_style)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for i in range(-step,step):
            index = index + 1
            z_styles = self.apply_rule(z,rule,i,scale,ls,le)

            img = self.generate_from_z(z = z_styles,
                                        truncation=truncation,
                                        normalize = False)

            name = '{}_step_{}.jpg'.format(id_style,index)
            save_path = os.path.join(output_dir,name).replace(os.sep,'/')

            img.save(save_path)
            np.save(os.path.splitext(save_path)[0],z_styles)

            self.storage.child(save_path).put(save_path)
            # self.storage.child(os.path.splitext(save_path)[0]+'.npy').put(os.path.splitext(save_path)[0]+'.npy')

    def apply_rule(self,z,rule,step,scale,ls,le):
        z = [z]*self.model.get_max_latents()

        for l in range(ls, le):
            z[l] = z[l] + rule * step * scale

        return z