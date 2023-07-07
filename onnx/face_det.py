import cv2
import os, time
import onnxruntime
from PIL import Image, ImageDraw
import numpy as np
try:
    from retinaface_ import cfg_mnet
    from retinaface_.prior_box import PriorBox
    from retinaface_.py_cpu_nms import py_cpu_nms
    from retinaface_.box_utils import decode, decode_landm
except:
    from .retinaface_ import cfg_mnet
    from .retinaface_.prior_box import PriorBox
    from .retinaface_.py_cpu_nms import py_cpu_nms
    from .retinaface_.box_utils import decode, decode_landm

facedet_model_path = "./detection_mobilenet0.25_Final.onnx"
ort_sess_options = onnxruntime.SessionOptions()
ort_sess_options.intra_op_num_threads = int(os.environ.get('ort_intra_op_num_threads', 0))
pwd = os.path.abspath(os.path.dirname(__file__))
ort_session = onnxruntime.InferenceSession(pwd+ facedet_model_path, sess_options=ort_sess_options)  # mobilenet
cfg = cfg_mnet



def margin_face(box, img_HW, margin=0.1):
    x1, y1, x2, y2 = [c for c in box]
    w, h = x2 - x1, y2 - y1
    new_x1 = max(0, x1 - margin*w)
    new_x2 = min(img_HW[1], x2 + margin * w)
    x_d = min(x1-new_x1, new_x2-x2)
    new_w = x2 -x1 + 2 * x_d  # 要保证脸左右两边都扩展相同的x_d个像素
    new_x1 = x1-x_d
    new_x2 = x2+x_d
    new_h = 1. * new_w   # 图像（112*112）宽高比是1.0
    if new_h>=h:
        y_d = new_h-h  # # 要保证脸上下两边都扩展相同的y_d的一半个像素
        new_y1 = max(0, y1 - y_d//2)
        new_y2 = min(img_HW[0], y2 + y_d//2)
    else:
        y_d = abs(new_h - h)  # # 要保证脸上下两边都缩减相同的y_d的一半个像素
        new_y1 = max(0, y1 + y_d // 2)
        new_y2 = min(img_HW[0], y2 - y_d // 2)
    # 由于图像人像可能靠近照片边缘，很有可能扩展到边缘就无法扩大。故此，宽度始终左右扩展相同，而高度可能不一定按1.0相对宽的比例扩的
    return list(map(int, [new_x1, new_y1, new_x2, new_y2]))


def detect_face(img, resize=1, confidence_threshold=0.97, top_k=30, nms_threshold=0.4, keep_top_k=15):
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    # if scale
    img -= np.array([104., 117., 123.]) # BGR
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    loc, conf, landms = ort_outs
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    prior_data = priorbox.forward()

    boxes = decode(loc.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale * resize
    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(landms.squeeze(0), prior_data, cfg['variance'])
    scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    landms = landms * scale1 * resize
    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    # keep top-K faster NMS
    # dets = dets[:keep_top_k, :4] # get rid of score
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    # dets = np.concatenate((dets, landms), axis=1)
    box_order = np.argsort((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))[::-1]
    dets = dets[box_order, :]
    landms = landms[box_order, :]

    # landms = np.reshape(landms, (landms.shape[0], 5, 2))
    # if 0 in dets.shape:
    #     return None, None
    # return dets, landms
    return np.concatenate((dets, landms), axis=1)



#
# if __name__ == '__main__':
#     path = r'E:\pro\face_get_server\test_data\test2'
#     path = r'C:\Users\Whty\Desktop\2\light'
#     files = [x for x in os.listdir(path)]
#     a = []
#     s =time.time()
#     for i, x in enumerate(files):
#         mat = cv2.cvtColor(cv2.imread(os.path.join(path, x)), cv2.COLOR_BGR2RGB)
#         dets, landms = detect_face(mat)
#         print(dets.shape)
#         print(landms.shape)
#         if len(dets)==0:
#             print(x)
#         a.append(len(dets))
#     print(time.time()-s)
#     print(len(list(set(a))))
#     print(a.count(0))
#     print(a.count(1))
#     print(a.count(2))


if __name__ == '__main__':
    # img_path = r"data/ai.jpg"
    img_path = r"data/55.png"
    # img_path = r"../test_data/2.png"
    img_path = r"E:\pro\pyqt/1000000004.jpg"
    # img_path = r"../img/22c4ac7eecd83332237ffa7abcaab42d.jpeg"
    # img_path = r"data/img.png"
    # img_path = r"data/55t.jpg"

    # read img
    img_name = os.path.basename(img_path)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)

    # det face landmark
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (320, 180))
    boxes, points = detect_face(rgb, 1)
    # rgb = cv2.resize(rgb, (320*2, 180*2))

    #
    # from face_align import norm_crop
    # res = norm_crop(img, points[0])
    # print(res.shape)
    # cv2.imshow('s',res)
    # cv2.waitKey(0)

    num = 0 if boxes is None else len(boxes)
    print(f'face num: {num} ')
    points = points.reshape((points.shape[0], -1, 2))
    # Draw boxes and save faces
    img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, (box, point) in enumerate(zip(boxes, points)):
        # if i>0:
        #     break
        w = box[2]-box[0]
        h = box[3]-box[1]
        print(w,h, box)
        print(point)

        margin_box = margin_face(box, (img.size[1], img.size[0]))
        w = margin_box[2] - margin_box[0]
        h = margin_box[3] - margin_box[1]
        print(w, h, margin_box)



        # draw
        draw.rectangle(box, width=2, outline=(255,0,0))
        draw.rectangle(margin_box, width=2, outline=(255,255,255))

        # face = img.crop(margin_box) # crop
        # face = face.resize((352,440), Image.LANCZOS) # resize
        # face.save(f'../data/{img_name}_detected_face_{i}.jpg')

        # keypoint
        for i,  p in enumerate(point):
            draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=2, outline=(255,255-51*i, 51*i))
    # save
    # img_draw.save(f'{img_name}_annotated_faces.jpg')
    # img_draw.show()
    cv2.imshow('sd', np.array(img_draw))
    cv2.waitKey(0)
