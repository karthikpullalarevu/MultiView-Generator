from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from PIL import Image
import os 
import torch 
import numpy as np 
import cv2 
import argparse
from simple_lama_inpainting import SimpleLama
from generate_3d_modified import RotateImage
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class DetectSegment:
    def __init__(self):
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.rotate_image = RotateImage()
        self.inpainting_lama = SimpleLama()
        self.threshold = 0.3
        detector_version = "IDEA-Research/grounding-dino-base"
        segmenter_version = "facebook/sam-vit-large"
        
        self.object_detector = pipeline(model=detector_version, task="zero-shot-object-detection", device=self.device) #initialize grounding dino
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_version).to(self.device) #initialize SAM 
        self.processor = AutoProcessor.from_pretrained(segmenter_version)
    
    def detect(self, image, labels, threshold, image_path):
        
        detections = self.object_detector(image,  candidate_labels=labels, threshold=threshold)
        try:
            result = detections[0]
        except IndexError:
            print("No object detected")
            return None
    
        bbox = [[[int(result['box']['xmin']), int(result['box']['ymin']), int(result['box']['xmax']), int(result['box']['ymax'])]]]
        
        #plot the bounding box
        image = np.array(image)
        x1, y1, x2, y2 = bbox[0][0]
        return bbox
    
    
    def segment(self, image, bbox):
        
        inputs = self.processor(images=image, input_boxes=bbox, return_tensors="pt").to(self.device)
        outputs = self.segmentator(**inputs)
        
        mask = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]
        mask = mask[0][0].squeeze().detach().cpu().numpy()
        return mask
    
    def apply_red_blend(self, original_image, mask, alpha=0.5):
        
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Ensure the mask is binary (0 or 1)
        binary_mask = mask.astype(bool)
        # Create a red color overlay
        red_color = np.zeros_like(original_image)
        red_color[binary_mask] = [255, 0, 0]  # Red color in BGR format

        # Blend the original image with the red overlay
        blended = cv2.addWeighted(original_image, 1, red_color, alpha, 0)
        return blended

    
    def paste_centered(self, background, foreground, bbox, center_x, center_y):
        # Calculate the size of the foreground image
        fg_width = bbox[2] - bbox[0]
        fg_height = bbox[3] - bbox[1]
        
        # print(fg_width, fg_height)
        # Calculate the top-left corner coordinates for pasting
        paste_x = center_x - fg_width // 2
        paste_y = center_y - fg_height // 2
        # print(paste_x,paste_y)
        # Create a new image with an alpha channel
        result = background.copy().convert('RGBA')
        
        # Paste the foreground onto the result
        result.paste(foreground, (paste_x, paste_y), foreground)
        return result

    def get_transparent_png(self, original_image,mask, bbox, background_image, first_bbox):
        # if isinstance(original_image, Image.Image):
        #     original_image = np.array(original_image)
        first_width = first_bbox[2] - first_bbox[0]
        first_height = first_bbox[3] - first_bbox[1]
        # print(first_width, first_height)
        #get_center of the bbox
        center_x = (first_bbox[0] + first_bbox[2]) // 2
        center_y = (first_bbox[1] + first_bbox[3]) // 2
        
        # print(center_x, center_y)
        
        #mask is as PIL image
        mask = np.array(mask)
        mask = (mask).astype(np.uint8) * 255
        alpha = mask
        if original_image.mode != 'RGBA':
            original_image = original_image.convert('RGBA')
        
        image_np = np.array(original_image)
        result = np.dstack((image_np[:,:,:3], alpha))
        #crop the image 
        x1, y1, x2, y2 = bbox
        result = result[y1:y2, x1:x2]
        bbox = [0, 0, first_width, first_height]
        result = cv2.resize(result, (first_width, first_height))
        transparent_image = Image.fromarray(result, 'RGBA')
       
        result = self.paste_centered(background_image, transparent_image, bbox, center_x, center_y)
        return result

    def apply_red_mask(self, original_image, mask, bbox, task = 'first'):
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Ensure the mask is binary (0 or 1)
        binary_mask = mask.astype(bool)
        #reverse the binary mask 
        # binary_mask = ~binary_mask
        
        white_image = np.ones_like(original_image) * 255
        white_image[binary_mask] = original_image[binary_mask]
        if task == 'first':
            x1, y1, x2, y2 = bbox
            if np.sum(binary_mask) < 6000 or x2-x1 < (1/3 * original_image.shape[1]) or y2-y1 < (1/3 * original_image.shape[0]):
            
                print('Cropping Image since object is too small')
           
                x1, y1, x2, y2 = bbox
                padding = 40
                x1 -= padding
                y1 -= padding
                x2 += padding
                y2 += padding
                # Get image dimensions
                img_height, img_width = white_image.shape[:2]
                # Ensure coordinates are within the image boundaries
                # Get image dimensions
                
                # Calculate the width and height of the padded bounding box
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Determine the size of the square to crop
                crop_size = max(bbox_width, bbox_height)
                
                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Calculate the new coordinates for the square crop
                new_x1 = center_x - crop_size // 2
                new_y1 = center_y - crop_size // 2
                new_x2 = new_x1 + crop_size
                new_y2 = new_y1 + crop_size
                
                # Ensure coordinates are within image boundaries
                new_x1 = max(new_x1, 0)
                new_y1 = max(new_y1, 0)
                new_x2 = min(new_x2, img_width)
                new_y2 = min(new_y2, img_height)
                
                # Crop the image
                white_image = white_image[new_y1:new_y2, new_x1:new_x2]
        
        # Set the pixels covered by the mask to red
        original_image[binary_mask] = [255, 0,0]  # Red color in BGR format
    
        return Image.fromarray(original_image), Image.fromarray(white_image)
    
    
    def generate_dilated_mask(self,mask):
        kernel = np.ones((50, 50), np.uint8)
        padded_mask = cv2.dilate(mask, kernel, iterations=1)
        return Image.fromarray(padded_mask)
    

    def process_image(self, image_path:str, label:str, task:str):
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
            
        labels = [label]
        labels = [label if label.endswith(".") else label+ "." for label in labels]
        
        bbox = self.detect(image, labels, self.threshold, image_path)
        mask = self.segment(image, bbox)
        padded_mask = self.generate_dilated_mask(mask.astype(np.uint8) * 255)
        red_mask,object_mask = self.apply_red_mask(image, mask, bbox[0][0], task)
        
        mask = mask.astype(bool)
        mask = Image.fromarray(mask)
        return image, bbox[0][0], mask, padded_mask, red_mask, object_mask

    def get_background(self, image, padded_mask):
        padded_mask = padded_mask.convert('L')
        background = self.inpainting_lama(image, padded_mask)
        return background
    
        
    def pipeline(self, image_path:str, label:str, output_path:str, do_rotation: bool = False, azimuth=0, polar=0):
        
        #process the image 
        image, bbox, mask, padded_mask, red_mask, object_mask = self.process_image(image_path, label, task = 'first')
        print('Detecting and Segmenting the object..')
        #save all the masks 
        
        if not do_rotation:
            print('Generating Mask..')
            red_mask.save(output_path)
            return
        else:
            print('Performing Rotation..')
           
                
            save_path = output_path #+ f"{i}_rotated.png"
            rotated_img = self.rotate_image.rotate_image_2d(object_mask, -azimuth, -polar)
           
            #process rotated_image as img_2 
            image_rotated, bbox_rotated, mask_rotated, padded_mask_rotated, red_mask_rotated, object_mask_rotated = self.process_image(rotated_img, label, task = 'second')
            
            background_image = self.get_background(image, padded_mask)
           
            
            transformed_image = self.get_transparent_png(image_rotated, mask_rotated, bbox_rotated, background_image, bbox)
            
            transformed_image.save(save_path)
            

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to the image")
parser.add_argument("--class_name", type=str, required=True, help="Text Prompt for the object to detect")
parser.add_argument("--output", type=str, required=True, help="Path to save the output image")
#add optional arguments
parser.add_argument("--azimuth", type=str, help="Aziuth angle for rotation")
parser.add_argument("--polar", type=str, help="Polar angle for rotation")

#check if user has provided the azimuth and polar angles

args = parser.parse_args()
do_rotation = False
if args.azimuth and args.polar:
    do_rotation = True
    azimuth = float(args.azimuth)
    polar = float(args.polar)

else:
    do_rotation = False
    azimuth = 0
    polar = 0

print(do_rotation, azimuth, polar)

detector = DetectSegment()
detector.pipeline(args.image, args.class_name, args.output, do_rotation, azimuth, polar)

                
