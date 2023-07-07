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

| input | torch output | onnx output |
| :-: |:-:| |:-:|
|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/cropped_faces/Justin_Timberlake_crop.png" height="80%" width="80%">|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/imgs/Justin_Timberlake_v2.jpg" height="80%" width="80%">|
|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/cropped_faces/Julia_Roberts_crop.png" height="80%" width="80%">|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/imgs/Julia_Roberts_v2.jpg" height="80%" width="80%">|
|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/cropped_faces/Paris_Hilton_crop.png" height="80%" width="80%">|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/imgs/Paris_Hilton_v2.jpg" height="80%" width="80%">|

