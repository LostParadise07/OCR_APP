import onnx
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
from auth_app import db
import math
from shapely.geometry import Polygon
import pyclipper
import os
from tqdm import tqdm
from auth_app import app
from auth_app.user.models import User, Message,ImageSegment
from auth_app.hindi_detection.recognizer_onnx import main
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def resize_image(img):
    height, width, _ = img.shape
    if height < width:
        new_height = 736
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = 736
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    
    resized_img = cv2.resize(img, (640,640))#new_width, new_height)
    return resized_img

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
    original_shape = img.shape[:2]
    img = resize_image(img)
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    img -= RGB_MEAN
    img /= 255.
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img, original_shape


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
            points[index_3], points[index_4]]
    return box, min(bounding_box[1])

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

def polygons_from_bitmap(pred, _bitmap, dest_width, dest_height):
    '''
    _bitmap: single map with shape (1, H, W),
        whose values are binarized as {0, 1}
    '''

    # assert _bitmap.size(0) == 1
    bitmap = _bitmap[0][0]  # The first channel
    pred = pred[0][0]

    print(pred.shape, bitmap.shape)
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours(
        (bitmap*255).astype(np.uint8),
        cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:1000]:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        
        
        score = box_score_fast(pred, points.reshape(-1, 2))
        if 0.55 > score:
            continue

        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        min_size = 0
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))

        # print(sside)
        if sside < min_size + 2:
            continue

        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        box[:, 0] = np.clip(
            np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(
            np.round(box[:, 1] / height * dest_height), 0, dest_height)

        boxes.append(box.tolist())
        scores.append(score)


    return boxes, scores

def run_inference_hindi(file,name,message):
    img_dir= os.path.join(app.config['UPLOAD_FOLDER'], name)

    # img_dir = os.listdir('val_images/')
    model_path = os.path.join(os.getcwd(),"auth_app/hindi_detection/","db_resnet18_new.onnx")
    onnx_model = onnx.load(model_path)
    node_list = []
    img_name=tqdm(img_dir)

    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(model_path)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    img, original_shape = load_image(img_path)
    batch = dict()
    batch['shape'] = [original_shape]
    batch['image'] = img
    
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    # print(ort_outs.shape, ort_outs)
    _, _, height, width = ort_outs.shape

    segmentation = ort_outs > 0.3
    # print(segmentation)
    boxes, _ = polygons_from_bitmap(
                    ort_outs,
                    segmentation, 640, 480)

    original_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_image = cv2.resize(original_image, (640,480))
    original_shape = original_image.shape
    pred_canvas = original_image.copy().astype(np.uint8)
    pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))
    texts=''
    for box in boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)
        
        x,y,w,h = cv2.boundingRect(box)
        # mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
        # cv2.fillPoly(mask, [box], (255))
        # masked_img = cv2.bitwise_and(original_image, original_image, mask=mask)
        cropped_img = original_image[y:y+h, x:x+w]
        # print("***********************************")
        # print(cropped_img)
        # print(box.shape)        
        imagename = 'uploads/' +str(box) + "_" + name
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],imagename), cropped_img)
        text = main(cropped_img)
        texts=texts+text
        imagesegment = ImageSegment(message_id=message.id, segment_image=imagename, text=text, text_modified=text)
        db.session.add(imagesegment)
        db.session.commit()


    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], name), pred_canvas)
    message.image_name=name
    message.text=texts
    message.text_modified=text
    db.session.commit()
