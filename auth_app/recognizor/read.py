import math
from argparse import Namespace
from PIL import Image

import torch
import torch.utils.data

from auth_app.recognizor.model import Model
from auth_app.recognizor.dataset import NormalizePAD
from auth_app.recognizor.utils import CTCLabelConverter, AttnLabelConverter
import os
def text_recognizer(img_cropped):
    """ opt configuration """
    opt = Namespace()
    opt.saved_model = os.path.join(os.getcwd(),"auth_app/recognizor/","best_norm_ED.pth")
    opt.batch_max_length = 100
    opt.imgH = 32
    opt.imgW = 400
    opt.rgb = False
    opt.FeatureExtraction = "UNet"
    opt.SequenceModeling = "DBiLSTM"
    opt.Prediction = "CTC"
    opt.input_channel = 1
    opt.output_channel = 512
    opt.hidden_size = 256
    if opt.FeatureExtraction == "HRNet":
        opt.output_channel = 32
    """ vocab / character number configuration """
    file = open(os.path.join(os.getcwd(),"auth_app/recognizor/","UrduGlyphs.txt"))
    content = file.readlines()
    content = ''.join([str(elem).strip('\n') for elem in content])
    opt.character = content+" "
    device = torch.device('cpu')
    opt.device = device
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    model = Model(opt)
    model = model.to(device)
    # load model


    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model.eval()
    
    """ Image processing """
    if opt.rgb:
        opt.input_channel = 3
        img = img_cropped.convert('RGB')
    else:
        img = img_cropped.convert('L')
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    img = img.resize((resized_w, opt.imgH), Image.BICUBIC)
    transform = NormalizePAD((1, opt.imgH, opt.imgW))
    img = transform(img)
    img = img.unsqueeze(0)
    batch_size = 1
    img = img.to(device)
    
    """ Prediction """
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    
    return preds_str

if __name__ == '__main__':
    image_path = "test.jpg"
    img_cropped = Image.open(image_path)
    preds_str = text_recognizer(img_cropped)
    print(preds_str)