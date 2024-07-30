# MultiView-Avataar_task
![Alt text](assets/images/workflow.png)
<p align = "center"> Fig 1. Workflow</p>


<p align="center">

  Here is a description of the final workflow:

**Step 1**: Run Grounding Dino + SAM (Grounding SAM) on the image by using the test prompt to extract the mask of the object. <br>
**Step 2**: Highlight the object by making all the other pixels white. <br>
**Step 3**: Run Zero Shot one image to 3D using the given Azimuth & Polar values. <br>
**Step 4**: Enhance the object size from Step 3 using EDSR model. Run Grounding Dino + SAM to get segment the object and apply alpha channel to make the background transparent. <br>
**Step 5**: From Step 1, pad the binary mask by using cv2 dilations with kernel size 50*50. Extract the background image by running SD Inpainting on the original image using the binary mask. <br>
**Step 6**: Using results from Step 5 and Step 4, we use the coordinates of object in original image to place the rotated object. <br>
</p>

<h2>Getting Started (Inference)</h2>
Expected time cost per image: 120s on an NVIDIA RTX A6000.(If dockerized, would be lower) <br>
Hardware Requirements: GPU VRAM >= 27GB; RAM >= 30 GB
<br><br>
<details>
<summary>Step 1: Create and activate a conda environment and install the dependencies. </summary> 
  
```bash
conda create -n multiview python=3.9
conda activate multiview
cd MultiView-Avataar_task
pip install -r requirements.txt
```
</details>

<details>
<summary>Step 2: Download the checkpoint for one shot image to 3D. </summary> 

```bash
cd MultiView-Avataar_task
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```
</details>

<details>
<summary>Step 3: To perform only segmenatation, do not pass the azimuth and polar values. </summary> 

```bash
python3 run.py --image ./laptop.jpg --class_name "laptop" --output ./generated.png
```
</details>

<details>
<summary>Step 4: To perform image rotation, use the following command: </summary> 

```bash
python3 run.py --image ./laptop.jpg --class_name "laptop" --output ./generated.png --azimuth +80 --polar +0
```

</details>


<h2>Experimentation & Approach</h2>
<details>
<summary>Step 1: Segmentation </summary> 
   <p>
     1. Started with ClipSeg as I had used it before. Tried different confidence values but the model wasnt performing well. 
     <br>
     <img src = "assets/images/clipseg.png" alt="clipseg results"
   </p>
      
</details>







