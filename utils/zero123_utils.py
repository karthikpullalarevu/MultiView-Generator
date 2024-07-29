import os
import numpy as np
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import CLIPImageProcessor
from torch import autocast
from torchvision import transforms
import math 
from super_image import EdsrModel, ImageLoader
from PIL import Image

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


def init_model(device, ckpt, half_precision):
    config = 'sd-objaverse-finetune-c_concat-256.yaml'
    config = OmegaConf.load(config)
    half_precision = True
    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    if half_precision:
        models['turncam'] = load_model_from_config(config, ckpt, device=device).half()
    else:
        models['turncam'] = torch.compile(load_model_from_config(config, ckpt, device=device))
    #print('Instantiating StableDiffusionSafetyChecker...')
    #models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
     #   'CompVis/stable-diffusion-safety-checker').to(device)
    models['clip_fe'] = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14")
    # We multiply all by some factor > 1 to make them less likely to be triggered.
    # models['nsfw'].concept_embeds_weights *= 1.2
    # models['nsfw'].special_care_embeds_weights *= 1.2

    return models

@torch.no_grad()
def sample_model_batch(model, sampler, input_im, xs, ys, n_samples=4, precision='autocast', ddim_eta=1.0, ddim_steps=75, scale=3.0, h=256, w=256):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = []
            # print(xs,ys)
            
            for x, y in zip(xs, ys):
                T.append([np.radians(x), np.sin(np.radians(y)), np.cos(np.radians(y)), 0])
            T = torch.tensor(np.array(T))[:, None, :].float().to(c.device)
            # print(c.shape,T.shape)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            # print('image: ', input_im.shape)
            cond['c_concat'] = [model.encode_first_stage(input_im).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            # print(cond['c_concat'][0].shape)
            print(model.encode_first_stage(input_im).mode().shape)
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            # print(shape)

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            # print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            ret_imgs = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            del cond, c, x_samples_ddim, samples_ddim, uc, input_im
            torch.cuda.empty_cache()
            return ret_imgs


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def extract_features(image, model):
    # image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image).flatten()
    return features


def check_best_matching_image(images, original_image):
    model = models.vgg16(pretrained=True).features
    model.eval()
    
    original_features = extract_features(original_image, model)
    generated_features = [extract_features(image, model) for image in images]
    distances = [torch.norm(original_features - gf).item() for gf in generated_features]
    best_match_index = np.argmin(distances)
    print('best_match_index:', best_match_index)
    return images[best_match_index]

    
@torch.no_grad()
def predict_stage1_gradio(model, raw_im, adjust_set=[], device="cuda", ddim_steps=75, scale=3.0, azimuth=0, polar=0):
    #resize the image to nearest multiple of 64 
    
    width, height = raw_im.size
    raw_im = raw_im.resize((256,256), Image.LANCZOS)
        
    # input_im_init = preprocess_image(models, raw_im, preprocess=False)
    input_im_init = np.asarray(raw_im, dtype=np.float32) / 255.0
    input_im = transforms.ToTensor()(input_im_init).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    adjust_set = [0]
    # stage 1: 8
    delta_x_1_8 = [polar]*4#[0] * 4 + [30] * 4 + [-30] * 4
    delta_y_1_8 = [azimuth]*4 #[0+90*(i%4) if i < 4 else 30+90*(i%4) for i in range(8)] + [30+90*(i%4) for i in range(4)]
   
    ret_imgs = []
    sampler = DDIMSampler(model)
    # sampler.to(device)
    if adjust_set != []:
        x_samples_ddims_8 = sample_model_batch(model, sampler, input_im, 
                                               [delta_x_1_8[i] for i in adjust_set], [delta_y_1_8[i] for i in adjust_set], 
                                               n_samples=len(adjust_set), ddim_steps=ddim_steps, scale=scale)
    # else:
    #     x_samples_ddims_8 = sample_model_batch(model, sampler, input_im, delta_x_1_8, delta_y_1_8, n_samples=len(delta_x_1_8), ddim_steps=ddim_steps, scale=scale)
    
    sample_idx = 0
    for stage1_idx in range(len(x_samples_ddims_8)):
        x_sample = 255.0 * rearrange(x_samples_ddims_8[sample_idx].numpy(), 'c h w -> h w c')
        out_image = Image.fromarray(x_sample.astype(np.uint8))
        ret_imgs.append(out_image)
        
        save_path = "/mnt/sravanth/"
        if save_path:
            out_image.save(os.path.join(save_path, f'{stage1_idx}.png'))
        sample_idx += 1

    # best_image = check_best_matching_image(ret_imgs, raw_im)
    best_image = ret_imgs[0].resize((width, height), Image.LANCZOS)
    edsr_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    best_image = ImageLoader.load_image(best_image)
    preds = edsr_model(best_image)
    ImageLoader.save_image(preds, './scaled_2x.png')
    preds = Image.open('./scaled_2x.png')
    
    #delete the scaled image
    os.remove('./scaled_2x.png')
    # preds = Image.fromarray()
    return [preds]
