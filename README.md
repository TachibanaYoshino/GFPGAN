# GFPGAN-ONNX
This is the onnxruntime inference code for  GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior (CVPR 2021). Official code: https://github.com/TencentARC/GFPGAN


## convert torch to onnx.
```
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

python torch2onnx.py  --src_model_path ./GFPGANv1.3.pth --dst_model_path ./GFPGANv1.3.onnx --img_size 512 
```

## run onnx demo.
```
python demo_onnx.py --model_path GFPGANv1.3.onnx --image_path ./cropped_faces/Adele_crop.png --save_path Adele_v3.jpg
```

| input | output|
| :-: |:-:|
|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/cropped_faces/Justin_Timberlake_crop.png" height="80%" width="80%">|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/imgs/Justin_Timberlake_v2.jpg" height="80%" width="80%">|
|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/cropped_faces/Julia_Roberts_crop.png" height="80%" width="80%">|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/imgs/Julia_Roberts_v2.jpg" height="80%" width="80%">|
|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/cropped_faces/Paris_Hilton_crop.png" height="80%" width="80%">|<img src="https://github.com/xuanandsix/GFPGAN-onnxruntime-demo/raw/main/imgs/Paris_Hilton_v2.jpg" height="80%" width="80%">|

