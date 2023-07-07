# GFPGAN-ONNX
This is the onnxruntime inference code for  GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior (CVPR 2021). Official code: https://github.com/TencentARC/GFPGAN


## convert gfpGAN to onnx.
```
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

python torch2onnx_gfpGAN.py  --src_model_path ./GFPGANv1.3.pth --dst_model_path ./GFPGANv1.3.onnx --img_size 512 
```

## convert retinaface to onnx.     
Modify gfpgan/utils.py    
1. line 85, det_model='retinaface_mobile0.25'    
2. line 92-121, uncomment   
3. Execute the following script.   
```
python inference_gfpgan.py   
```

## run onnx demo.
```
python onnx_infer.py  --image_path ../inputs/whole_imgs --output results
```

Input | torch output | onnx output
:------:  | :------: | :------:
<img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/cropped_faces/10045_01.png" height="50%" width="50%"> | <img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/cropped_faces/10045_01.png" height="50%" width="50%"> | <img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/restored_faces/10045_01.png" height="50%" width="50%">
<img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/cropped_faces/10045_00.png" height="50%" width="50%"> | <img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/cropped_faces/10045_00.png" height="50%" width="50%"> | <img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/restored_faces/10045_00.png" height="50%" width="50%">
<img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/cropped_faces/Blake_Lively_01.png" height="50%" width="50%"> | <img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/cropped_faces/Blake_Lively_01.png" height="50%" width="50%"> | <img src="https://github.com/TachibanaYoshino/GFPGAN/blob/onnx/onnx/results/restored_faces/Blake_Lively_01.png" height="50%" width="50%">
      
# Acknowledgement
Thanks to [GFP-GAN](https://github.com/TencentARC/GFPGAN), [GFPGANv1.3-to-ncnn](https://github.com/magicse/GFPGANv1.3-to-ncnn), [GFPGAN-onnxruntime-demo](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo) for sharing their code.
