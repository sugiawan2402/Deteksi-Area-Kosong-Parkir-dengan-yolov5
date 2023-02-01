import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from roi import *
import pyautogui
@torch.no_grad()
def mouseClick(events, x, y, flags, params):
    posList = []
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
                print(posList)

def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / '0',  # file/dir/URL/glob, 0 for webcam
        imgsz=(720, 1280),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)

):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Dataloader
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
        #xyxy
        # cv2.rectangle(im0, (30, 265), (310, 510), (255, 255, 0), 3)  # biru muda
        # cv2.rectangle(im0, (300, 265), (550, 500), (0, 255, 255), 3)  # kuning
        # cv2.rectangle(im0, (595, 265), (880, 500), (0, 0, 255), 3)  # merah
        # cv2.rectangle(im0, (915, 265), (1250, 510), (255, 0, 0), 3)  # biru

        
        
    lot = [
        [3, 150, 150, 360],
        [158, 150, 310, 360],
        [318, 150, 480, 360],
        [487, 150, 638, 360],
        # [305, 180, 625, 360],
          # [30, 265, 315, 515],  # 1
          # [300, 265, 550, 515],  # 2
          # [595, 265, 880, 515],#3
          # [900, 265, 1270, 515]#4
           ]
    # Run inference
    while True:
        object = []
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
    
            # Inference
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
    
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
    
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
             
                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                # cv2.rectangle(im0, (595, 265), (880, 500), (0, 0, 255), 3)#merah
                # cv2.rectangle(im0, (300, 265), (550, 500), (0, 255, 255), 3)#kuning
                # cv2.rectangle(im0, (915, 265), (1250, 510), (255, 0, 0), 3) #biru
                # cv2.rectangle(im0, (30, 265), (310, 510), (255, 255, 0), 3)  #biru muda
                # cv2.circle(im0, (595, 265), 2, 255, 3)
                # cv2.circle(im0, (880, 500), 2, 255, 3)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        
                    # bus : 6, car : 3, truck : 8
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # print(cls)
                        c = int(cls)  # integer class
                        conf = conf * 100
                        label = f'{names[c]} {conf:.2f} %'
                        
                        if names[c] == 'car':
                            annotator.box_label(xyxy, label, color=colors(c, True))
                             #print(int (xyxy[0]), int (xyxy[1]), int (xyxy[2]), int (xyxy[3]))
                             #cv2.rectangle(im0, (int(xyxy[0]) + 20, int(xyxy[1]) + 50),
                              #             (int(xyxy[2]) - 50, (int (xyxy[3]) - 50)), (0,0,255), 3)
                            x = int(xyxy[0])
                            y = int(xyxy[1])
                            h = int(xyxy[2])
                            w = int(xyxy[3])
                            print(x,y,h,w)
                            object.append([x + 10 ,y+10,h-10,w-10])    
                    
                show_roi_area(im0, object, lot, line_thickness=3)
                im0 = annotator.result()
    
                    # cv2.waitKey(1)  # 1 millisecond
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                
                
                cv2.imshow(str(p), cv2.resize(im0, (640, 480), interpolation=cv2.INTER_AREA))
                # cv2.imshow(str(p), im0)
                cv2.waitKey(1)
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[720], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
  
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":

    opt = parse_opt()
    main(opt)
    
