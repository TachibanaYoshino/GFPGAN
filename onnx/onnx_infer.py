import argparse
import cv2
import glob
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
import faceRestoreHelper

device_name = ort.get_device()
print(device_name)
providers=None
if device_name == 'CPU':
    providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    providers = ['CUDAExecutionProvider'] #, 'CPUExecutionProvider']

gfp_model_path = "./GFPGANv1.4_sim.onnx"
ort_sess_options = ort.SessionOptions()
ort_sess_options.intra_op_num_threads = int(os.environ.get('ort_intra_op_num_threads', 0))
pwd = os.path.abspath(os.path.dirname(__file__))
ort_session = ort.InferenceSession(os.path.join(pwd, gfp_model_path), sess_options=ort_sess_options, providers=providers)  # mobilenet


def argsparse():
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('-i', '--input',type=str,default='/data/xinchen/update/deepface/roop/results/Kobe_swapFace.mp4', help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument('-c', '--save_croped', type=bool, default=True, help='save the croped face')
    parser.add_argument('-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('-p', '--use_parse', type=bool, default=True, help='Segmentation face. Default: True')
    parser.add_argument('-sr', '--realesrgan', choices=['', 'realesrgan'], default="", help='background upsampler. Default: realesrgan')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--ext',type=str,default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    args = parser.parse_args()
    return args


def gfpgan(img, upscale=2, use_parse=True, realesrgan=""):
    img = faceRestoreHelper.read_image(img)
    det_faces, all_landmarks_5 = faceRestoreHelper.get_face_landmarks_5(img)
    cropped_faces, affine_matrices = faceRestoreHelper.align_warp_face(img, all_landmarks_5, face_size=512)
    # face restoration
    restored_faces = []
    for cropped_face in cropped_faces: # 512*512 aligned faces
        x = faceRestoreHelper.preprocess(cropped_face)
        # feedforward
        pred = ort_session.run(None, {ort_session.get_inputs()[0].name: x})[0]
        restored_face = faceRestoreHelper.post_processing(pred)
        restored_faces.append(restored_face)

    if upscale > 1:
        if realesrgan:
            img = realesrgan.enhance(img, outscale=upscale)[0]
        else:
            h, w = img.shape[:2]
            h_up, w_up = int(h * upscale), int(w * upscale)
            img = faceRestoreHelper.icv_resize(img, (w_up, h_up))

    inverse_affine_matrices = faceRestoreHelper.get_inverse_affine(affine_matrices, upscale_factor=upscale)
    restored_img = faceRestoreHelper.paste_faces_to_image(img, restored_faces, inverse_affine_matrices, upscale_factor=upscale, use_parse=use_parse)
    return cropped_faces, restored_faces, restored_img


def image_enforce(args):
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)
    upscale = args.upscale
    use_parse = args.use_parse
    realesrgan = args.realesrgan
    if realesrgan:
        from inference_realesrgan_ByOnnx import RealESRGANer

        esr_session = ort.InferenceSession("./realesr-general-x4v3.onnx", providers=providers)
        realesrgan = RealESRGANer(scale=4, ort_session=esr_session, tile=400, tile_pad=10, pre_pad=0)
    # ------------------------ restore ------------------------
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = gfpgan(input_img, upscale, use_parse, realesrgan)
        # save faces
        if args.save_croped:
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(args.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
                faceRestoreHelper.imwrite(cropped_face, save_crop_path)
                # save restored face
                if args.suffix is not None:
                    save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
                else:
                    save_face_name = f'{basename}_{idx:02d}.png'
                save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
                faceRestoreHelper.imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                faceRestoreHelper.imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if args.ext == 'auto':
                extension = ext[1:]
            else:
                extension = args.ext
            if args.suffix is not None:
                save_restore_path = os.path.join(args.output, 'restored_imgs',
                                                 f'{basename}_{args.suffix}.{extension}')
            else:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
            faceRestoreHelper.imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{args.output}] folder.')


def video_enforce(args):
    if not args.input.endswith('.mp4'):
        raise "input error"
    os.makedirs(args.output, exist_ok=True)
    upscale = args.upscale
    use_parse = args.use_parse
    realesrgan =  args.realesrgan

    if realesrgan:
        from inference_realesrgan_ByOnnx import RealESRGANer
        esr_session = ort.InferenceSession("./realesr-general-x4v3.onnx", providers=providers)
        realesrgan = RealESRGANer(scale=4, ort_session=esr_session,tile=400,tile_pad=10,pre_pad=0)

    vid = cv2.VideoCapture(args.input)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    pbar = tqdm(total=total )
    pbar.set_description(f"Running: {os.path.basename(args.input)}")
    video_out = cv2.VideoWriter(os.path.join(args.output,os.path.basename(args.input)), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
    while True:
        # print("get me")
        ret, frame = vid.read()
        if not ret:
            break
        cropped_faces, restored_faces, restored_img = gfpgan(frame, upscale, use_parse, realesrgan)
        restored_img = faceRestoreHelper.icv_resize(restored_img,  (width, height))
        video_out.write(restored_img)
        pbar.update(1)
    pbar.close()
    video_out.release()
    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    args = argsparse()
    # ------------------------ input & output ------------------------
    if not args.input.endswith('.mp4'):
        image_enforce(args)
    else:
        video_enforce(args)




