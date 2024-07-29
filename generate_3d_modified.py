import sys 
import os 
sys.path.append('/mnt/sravanth/Adhoc/One-2-3-45')
from utils.zero123_utils import init_model, predict_stage1_gradio #, zero123_infer
import os 
# from elevation_estimate.estimate_wild_imgs import estimate_elev
import numpy as  np 
import torch 
from utils.utils import image_preprocess_nosave, gen_poses


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class RotateImage:
    def __init__(self):
        self.GPU_INDEX = 0
        self.device = f'cuda:{self.GPU_INDEX}'
        path_ckpt = "./zero123-xl.ckpt"
        self.models = init_model(self.device,path_ckpt , half_precision=True)


    def main_run(self, raw_im, scale, ddim_steps, azimuth, polar):
       model = self.models['turncam']
       results = predict_stage1_gradio(model, raw_im, adjust_set=list(range(4)), device=self.device, ddim_steps=ddim_steps, scale=scale, azimuth=azimuth, polar=polar)
       return results
                
    def rotate_image_2d(self, image, azimuth, polar):
        
        # raw_im = image.convert('RGBA') #Image.open(image_path).convert('RGBA')
        results = self.main_run(image, scale = 3.0, ddim_steps=75, azimuth = azimuth, polar = polar)
        result = results[0]
        
        return result
