#!/2019/04:15 18:00
#!/2019/04:16 01:02
#!/2019/04:16 21:10

## A simple class FCOS, mainly to `inference_single_cvimage`.
"""
## A simple class FCOS, mainly to `inference_single_cvimage`.
cd FCOS/demo
python inference_single_cvimage.py
python inference_single_cvimage.py --config-file ../configs/fcos/fcos_R_50_FPN_1x.yaml --weight-file ../models/FCOS_R_50_FPN_1x.pth
"""

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir + "/..")
sys.path.insert(0, curdir)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import math
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from torchvision import transforms as T
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.rotation_box import RBoxList
from maskrcnn_benchmark.structures.rboxlist_ops import boxlist_nms


import argparse

save_path = '/media/liesmars/b71625db-4194-470b-a8ab-2d4cf46f4cdd/Object_detection/FCOS_pytorch/RFCOS/training_dir/test'
# for dota
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
# for hrsc2016
# wordname_15 = ['pl', 'bb', 'bre', 'gf', 'sv', 'lv', 'ship', 'te',
#                'bk', 'st',  'sb', 'rb', 'hb', 'sp', 'hc']

textcolor = (5, 5, 245)

object_sizes_of_interest = np.array([
                [0, 64],#100
                [32, 128],#50
                [64, 256],#25
                [128, 512],#13
                [256, 800],#7
            ])

coco_names = ['back_ground'] + wordname_15

def draw_rotate_box_cv(img, boxes, labels, scores, level, response_point, end_points):
    # img = img + np.array(cfgs.PIXEL_MEAN)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    # img = np.array(img*255/np.max(img), np.uint8)
    end_points = end_points.astype(np.int32)

    num_of_object = 0
    for i, box in enumerate(boxes):
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
        if max(w,h)/(min(w,h)+0.1) > 18:
                continue

        label = labels[i]
        if label != 0:
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            # 5,5,255 
            color = [5,255,5]#color_15[label]

            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)

            r=4
            thickness = 2
            if w*h > 5000:
                r=10
                thickness = 4 
            cv2.drawContours(img, [rect], -1, color, thickness, lineType=4)

            category = label#str(label)

            if scores is not None:
                l = int(level[i])
                
                cv2.circle(img,(int(response_point[i][0]),int(response_point[i][1])),r,(0,0,230),-1)

                r=5
                if w*h > 5000:
                    r=10
                
                ptStart = (end_points[i][0], end_points[i][1])
                ptEnd = (end_points[i][2], end_points[i][3])
                # point_color = (0, 255, 0) # BGR
                
                lineType = 4
                cv2.line(img, ptStart, ptEnd, (200,0,200), thickness, lineType)

            else:
                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c + 40, y_c + 15),
                              color=color,
                              thickness=-1)
                cv2.putText(img,
                            text=category,
                            org=(x_c, y_c + 10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    return img

def create_colors(len=1):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len)]
    colors = [(int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255)) for rgba in colors]
    return colors

def crop_image(img, new_width, new_height, width_step, height_step, width, height, crop_size, step_size):
    crop_img = []
    if((new_width+crop_size<=width)&(new_height+crop_size<=height)):
        crop_img = img[new_height:(new_height+crop_size), new_width:(new_width+crop_size)]
    elif((new_width+crop_size>width)&(new_height+crop_size<=height)):
        crop_img = img[new_height:(new_height+crop_size), new_width:(width)]
    elif((new_width+crop_size<=width)&(new_height+crop_size>height)):
        crop_img = img[new_height:(height), new_width:(new_width+crop_size)]
    else:
        if((height-new_height>=(50))&((width-new_width>=(50)))):
            crop_img = img[new_height:(height), new_width:(width)]
    return crop_img

def write_dotaResult(image_name, boxes, scores, categories):
    word = image_name.split('/')
    word0 = word[-1]
    filename = word0.split('.')
    # for cls_id in range(1, cfgs.CLASS_NUM+1):
    for cls_id in range(len(coco_names)):
        if(cls_id == 0):
            continue
        cls = coco_names[cls_id]
        
        f = open(save_path + "/"+cls+".txt", 'a+')
        index = categories == float(cls_id)
        tem_boxes = boxes[index]
        tem_scores = scores[index]
        
        tem_categories = categories[index]
        for i in range(len(tem_scores)):
            boxes0 = tem_boxes[i]
            x_c, y_c, w, h, theta = boxes0[0], boxes0[1], boxes0[2], boxes0[3], boxes0[4]
            if max(w,h)/(min(w,h)+0.1) > 18:
                continue
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            # print(rect.reshape([-1]))
            rect = np.where(rect<=0, 1, rect)
            x1, y1, x2, y2, x3, y3, x4, y4 = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2][0], rect[2][1], rect[3][0], rect[3][1]
            # print(rect.reshape([-1]))
            # print(x1, y1, x2, y2, x3, y3, x4, y4)
            f.write(filename[0] + ' ')
            f.write(str(tem_scores[i]) + ' ')
            f.write(str(float(int(x1))) + ' ')
            f.write(str(float(int(y1))) + ' ')
            f.write(str(float(int(x2))) + ' ')
            f.write(str(float(int(y2))) + ' ')
            f.write(str(float(int(x3))) + ' ')
            f.write(str(float(int(y3))) + ' ')
            f.write(str(float(int(x4))) + ' ')
            f.write(str(float(int(y4))) + '\n')
        f.close()

def parse_args():
    parser = argparse.ArgumentParser(description="FCOS Object Detection Demo on OpenCV Image")
    parser.add_argument(
        "--config-file",
        default="../configs/fcos/fcos_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight-file",
        default="../models/FCOS_R_50_FPN_1x.pth",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=1,#"cuda:1",
        help="Device, default cuda:0",
    )

    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )

    parser.add_argument(
        "--image",
        type=str,
        default="man_dog.jpg",
        help="your test image path",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./"
    )

    parser.add_argument(
        "--start_imgid",
        type=str
    )
    

    args = parser.parse_args()
    return args


## A simple class FCOS, main to `inference_single_cvimage`.
class FCOS():

    coco_colors = create_colors(len(coco_names))

    def __init__(self, cfg, min_image_size=400):
        self.device = cfg.MODEL.DEVICE
        self.model = None
        self.cfg = cfg
        self.transform = self.build_transform(min_image_size)
        self.build_and_load_model()

    def build_and_load_model(self):
        cfg = self.cfg
        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir = cfg.OUTPUT_DIR)
        _ = checkpointer.load(cfg.MODEL.WEIGHT )
        model.eval()
        self.model = model

    def build_transform(self, min_image_size=400):
        cfg = self.cfg
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        transform = T.Compose(
            [
                T.ToPILImage(),
                # T.Resize(min_image_size),
                T.ToTensor(),
                T.Lambda(lambda x: x * 255),
                normalize_transform,
            ]
        )
        return transform

    def inference_single_cvimage(self, img, verbose=True, score_th=0.05, idx=0):
        def conver4xyTOxywha(boxes):
            result = np.empty((0, 5))
            for i in range(boxes.shape[0]):
                rect = boxes[i, :]
                box = np.int0(rect)
                box = box.reshape([4, 2])
                rect1 = cv2.minAreaRect(box)
                x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
                
                result = np.append(result, np.array([[x, y, w, h, theta]]), axis = 0)
            return result

        nh, nw = img.shape[:2]
        image = self.transform(img)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        ## compute prediction
        with torch.no_grad():
            predictions = self.model(image_list)
        

        # (h, w) 0 只有一张图片
        box_result = predictions[0].to("cpu")
        ratex, ratey =  nw / min_image_size, nh / min_image_size
        
        # pt_x,pt_y,x1,y1,x2,y2,h
        # print(nh, nw)
        box = box_result.bbox * torch.tensor([ratex])
        np_box_result = np.array(box)
        box = np.empty((0, 8))
        response_point = np.empty((0, 2))
        end_point = np.empty((0, 4))

        valid_ind = torch.ones([np_box_result.shape[0]], dtype = torch.long)
        for i in range(np_box_result.shape[0]):
            per_box = np_box_result[i, :]
            #pt_x pt_y for visualization of paper 
            pt_x,pt_y,x1,y1,x2,y2,h = per_box
            x_c, y_c=(x1+x2)/2, (y1+y2)/2
            w = math.sqrt(math.pow(x1-x2,2)+ math.pow(y1-y2,2))

            long_w, short_h = w,h
            a_x, a_y, b_x, b_y = x1,y1, x2,y2

            # *****************visualization of paper 
            # a_pt, a_b = [0,0], [0,0];
            # a_pt[0] = pt_x - a_x;
            # a_pt[1] = pt_y - a_y;
            # a_b[0] = b_x - a_x;
            # a_b[1] = b_y - a_y;
            # pt_proj_ab_length = (a_pt[0]*a_b[0] + a_pt[1]*a_b[1])/math.sqrt(a_b[0]*a_b[0]+a_b[1]*a_b[1]);

            # dst2pt1 = math.sqrt(a_pt[0]*a_pt[0]+a_pt[1]*a_pt[1])
            # if pt_proj_ab_length<=0 or dst2pt1 <= pt_proj_ab_length:
            #     temp = 0
            # else: 
            #     # print(math.sqrt(a_pt[0]*a_pt[0]+a_pt[1]*a_pt[1]), pt_proj_ab_length)
            #     pt_2_ab_length = math.sqrt(a_pt[0]*a_pt[0]+a_pt[1]*a_pt[1] - pt_proj_ab_length*pt_proj_ab_length+0.0001);
            #     # // ori centerness
            #     temp = min(pt_proj_ab_length, long_w - pt_proj_ab_length)/max(pt_proj_ab_length, long_w - pt_proj_ab_length)*\
            #                 min(short_h/2. - pt_2_ab_length, short_h/2. + pt_2_ab_length)/max(short_h/2. - pt_2_ab_length, short_h/2. + pt_2_ab_length);
            #     # print(temp)
            #     temp = max(0, temp)
            # *****************debug for centerness
                

            # 笛卡尔坐标系转为图像坐标系
            theta = math.atan2(y1-y2, x1-x2)
            theta = theta if theta <= 0 else (theta - math.pi)
            if theta <= -math.pi/2:
                w,h = h,w  
                theta = theta + math.pi/2


            # x_c, y_c, w, h, theta = per_box
            rect = ((x_c, y_c), (w, h), theta/math.pi*180)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            
            points = np.array([[rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2][0], rect[2][1], rect[3][0], rect[3][1]]])            
            box = np.append(box, points, axis = 0)
            response_point = np.append(response_point, np.array([[pt_x,pt_y]]), axis=0)
            end_point = np.append(end_point, np.array([[x1,y1,x2,y2]]),axis=0)
        box = torch.tensor(conver4xyTOxywha(box))

        valid_ind = valid_ind > 0
        label = box_result.get_field('labels')[valid_ind]

        score = box_result.get_field('scores')[valid_ind]
        level = box_result.get_field("levels")[valid_ind]
        # print(score)
        return score, label, box, response_point, level, end_point


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.device)

    save_path = os.path.join(save_path, args.save_dir)

    score_th = args.confidence_threshold
    min_image_size = args.min_image_size


    # update cfg
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHT = args.weight_file
    cfg.TEST.IMS_PER_BATCH = 1  # only test single image
    cfg.freeze()

    fcos = FCOS(cfg, min_image_size)

    with open(args.image,'r') as rf:
        k=0
        id=0
        for line in rf:
            k= k+1
            img_name = line.strip().split("/")[-1]
            if "P" in args.start_imgid and (args.start_imgid not in img_name) or \
               "P" not in args.start_imgid and (k<int(args.start_imgid)): 
                continue
            print(img_name)

            crop_size = 800
            # crop_size = 600
            step_size = 200
            raw_img1 = cv2.imread(line.strip(), cv2.IMREAD_COLOR)
            height, width = raw_img1.shape[0], raw_img1.shape[1]

            path = line.strip()
            word = path.split('/')
            filename = word[-1]
            
            
            bboxes = np.empty((0, 5))
            scores = np.empty(0)
            labels = np.empty(0)
            level = np.empty(0)
            response_point= np.empty((0, 2))
            end_points = np.empty((0, 4))
            
            if((width <= (crop_size))&(height <= (crop_size))):
                # raw_img1 = cv2.resize(raw_img1, (800, 841))
                raw_img1 = cv2.copyMakeBorder(raw_img1, 0, crop_size - raw_img1.shape[0], 0, crop_size - raw_img1.shape[1], cv2.BORDER_CONSTANT, value = (255, 255, 255))     
                # _scores, _labels, _bboxes, _response_point, _level
                
                return_result =  fcos.inference_single_cvimage(raw_img1, verbose=True, score_th = score_th)
                _scores, _labels, _bboxes, _response_point, _level, _endpoints = np.array(return_result[0]), np.array(return_result[1]),\
                     np.array(return_result[2]), np.array(return_result[3]), np.array(return_result[4]),np.array(return_result[5])
                
                
                # det_detections_r = draw_rotate_box_cv(raw_img1,
                #                                           boxes=box,
                #                                           labels=label,
                #                                           scores=score)
                # cv2.imwrite(os.path.join(save_path, filename), det_detections_r)
                # write_dotaResult(filename, box, score, label)
                # _scores, _labels, _bboxes = np.array(_scores), np.array(_labels), np.array(_bboxes)

                if(len(_scores)<=0):
                    continue

                bboxes = np.append(bboxes, _bboxes, axis = 0)
                scores = np.append(scores, _scores, axis = 0)
                labels = np.append(labels, _labels, axis = 0)   
                level  = np.append(level, _level, axis = 0)   
                response_point = np.append(response_point, _response_point, axis = 0)
                end_points  = np.append(end_points, _endpoints, axis = 0) 
                
                
            else:
                # raw_img = cv2.imread(a_img_name)
                # start = time.time()
                width_num  = math.ceil((width  - step_size)/(crop_size - step_size))
                height_num = math.ceil((height - step_size)/(crop_size - step_size))
                for width_step in range(width_num):
                    for height_step in range(height_num):
                        
                        new_width = width_step * crop_size - (width_step)*step_size
                        new_height = height_step * crop_size - (height_step)*step_size
                        crop_img = crop_image(raw_img1, new_width, new_height, width_step, height_step, width, height, crop_size, step_size)
                        if len(crop_img) == 0:
                            continue
                        if((crop_img.shape[0]<=1)|(crop_img.shape[1]<=1)):
                            continue
                        crop_img = cv2.copyMakeBorder(crop_img, 0, crop_size - crop_img.shape[0], 0, crop_size - crop_img.shape[1], cv2.BORDER_CONSTANT, value = (255, 255, 255))     
  

                        # start_time = time.time()
                        _scores, _labels, _bboxes, _response_point, _level, _endpoints = fcos.inference_single_cvimage(crop_img, verbose=True, score_th = score_th)
                        

                        _response_point[:,0] += new_width
                        _response_point[:,1] += new_height
                        _scores, _labels, _bboxes, _response_point, _level, _endpoints = \
                            np.array(_scores), np.array(_labels), np.array(_bboxes), \
                                np.array(_response_point), np.array(_level), np.array(_endpoints)
                        
        
                        _endpoints = _endpoints + np.array([[new_width, new_height,new_width, new_height]])


                        if(len(_scores)<=0):
                            continue
                        bboxes = np.append(bboxes, _bboxes+[new_width, new_height, 0, 0, 0], axis = 0)
                        scores = np.append(scores, _scores, axis = 0)
                        labels = np.append(labels, _labels, axis = 0)  
                        level  = np.append(level, _level, axis = 0)   
                        response_point = np.append(response_point, _response_point, axis = 0)
                        end_points  = np.append(end_points, _endpoints, axis = 0) 
                        
                                
            target = RBoxList(bboxes, raw_img1.shape, mode="xywha") 
            target.add_field("labels", torch.tensor(labels))
            target.add_field('score', torch.tensor(scores))
            target.add_field('response_point', torch.tensor(response_point))
            target.add_field('end_points', torch.tensor(end_points))
            target.add_field('levels', torch.tensor(level))
            target.to('cuda')
            

            target = boxlist_nms(target, 0.05)
            # print(os.path.join("./demo/dota_result/", filename))

            write_dotaResult(filename, np.array(target.bbox), np.array(target.get_field('score')), np.array(target.get_field('labels')))
            det_detections_r = draw_rotate_box_cv(raw_img1,
                                                        boxes=np.array(target.bbox),
                                                        labels=np.array(target.get_field('labels')),
                                                        scores=np.array(target.get_field('score')),
                                                        level=np.array(target.get_field('levels')),
                                                        response_point=np.array(target.get_field('response_point')),
                                                        end_points=np.array(target.get_field('end_points')))
            cv2.imwrite(os.path.join(save_path, filename), det_detections_r)#"square_"+
