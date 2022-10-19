import os
import argparse
import concurrent.futures
from preprocess.preprocessing import PreProcess
from postprocess.postprocessing import PostProcess
import shutil
import time
import tqdm
import numpy as np
import cv2
import torch
from datetime import datetime
from yolov7.utils.dataloader import CustomImageDataset
from torch.utils.data import DataLoader
from preprocess.slicing import CropFast
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression,  xyxy2xywh, set_logging
from yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel


class YOLT7:
    def __init__(self, opt):
        self.save_crops = opt.save_crops
        self.crop_size = opt.crop_size
        self.step_size = opt.step_size
        self.second_conf_thresh = opt.second_conf_thresh
        self.input = opt.input
        self.final_results = str(datetime.now().time()) + '/'
        self.crops_list_file = opt.crops_list_file
        self.cropfast = CropFast(self.input, self.save_crops, self.crop_size, self.step_size)
        self.preprocess = PreProcess(self.save_crops, self.crops_list_file)
        self.postprocess = PostProcess('yolov7/tmp_data/results.txt', self.final_results ,self.crop_size, self.second_conf_thresh )

        if os.path.exists('yolov7/tmp_data/'):
            shutil.rmtree('yolov7/tmp_data/')
            os.mkdir('yolov7/tmp_data/')
        else:
            os.mkdir('yolov7/tmp_data/')

        if os.path.exists(opt.save_crops):
            shutil.rmtree(opt.save_crops)
            os.mkdir(opt.save_crops)
        else:
            os.mkdir(opt.save_crops)


        if os.path.exists(opt.save_crops):
            shutil.rmtree(opt.save_crops)
            os.mkdir(opt.save_crops)
        else:
            os.mkdir(opt.save_crops)

        os.mkdir(self.final_results)


    def detect(self, opt, save_img=False):
        class_names = opt.names
        weights, view_img, imgsz, trace =  opt.weights, opt.view_img, opt.img_size, not opt.no_trace
        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, imgsz)

        if half:
            model.half()  # to FP16
        customdataset = CustomImageDataset(opt.crops_list_file)
        dataset = DataLoader(customdataset,
                             batch_size=2,
                             num_workers=4,
                             pin_memory=True)
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        with open('yolov7/tmp_data/results.txt', 'w') as f:
            for img, names in dataset:
                im0 = img[0]
                im0 = np.transpose(im0, (1, 2, 0))
                img = img.to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32

                # Warmup
                if device.type != 'cpu' and (
                        old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t3 = time_synchronized()

                # Process detections
                for det, name in zip(pred, names):  # detections per image
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            x,y,w,h = [*xywh]
                            x1 = int((x - w / 2) * opt.crop_size)
                            x2 = int((x + w / 2) * opt.crop_size)

                            y1 = int((y - h / 2) * opt.crop_size)
                            y2 = int((y + h / 2) * opt.crop_size)
                            if len(class_names)==1:
                                l = [name, class_names[0], conf.item(), x1, y1, x2, y2]
                            else:
                                l = [name, class_names[int(cls.item())], conf.item(), x1, y1, x2, y2]

                            l = ' '.join(str(e) for e in l)
                            f.write(l + '\n')

    def main(self, opt):
        self.cropfast.slicing_patches()
        self.preprocess.generate_valid_list()
        ts = time.time()
        with torch.no_grad():
            self.detect(opt)
        print(f'Total Detection Time : {time.time()-ts}s')
        global_df = self.postprocess.globalize_coordinates()
        name_groups = self.postprocess.separate_image_results(global_df)
        tnmsp1 =time.time()
        images_results_list = [(name, image_bboxes) for name, image_bboxes in name_groups]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.nms_thread, images_results_list)
        tnmsp2 = time.time()
        print(f' Total Plotting and NMS time: {tnmsp2 - tnmsp1}s')

    def nms_thread(self , image_name_bboxes):
        name, image_bboxes = image_name_bboxes
        img = cv2.imread(self.input + name + '.jpg')
        df_result = self.postprocess.nms(image_bboxes)
        self.postprocess.plot(img,name, df_result)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input/', help='path to the test input images')
    parser.add_argument('--save_crops', type=str, default='yolov7/tmp_data/output/', help='path to sliced images')
    parser.add_argument('--crop_size', type=int, default=640, help='size of crop')
    parser.add_argument('--step_size', type=int, default=500, help='stride for slicing')
    parser.add_argument('--crops_list_file', type=str, default='yolov7/tmp_data/images_list.txt')
    parser.add_argument('--final_results', type=str, default='result/')
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov7/ost.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--second_conf_thresh', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    #parser.add_argument('--names', type=list, default=['human', 'sup-board','boat','bouy' ,'sailboat', 'kayak'] , help='list of class names')
    parser.add_argument('--names', type=list, default=['OST'],
                        help='list of class names')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    t1 = time.time()
    yolt7 = YOLT7(opt)
    yolt7.main(opt)
    print(f' Total time taken {time.time()-t1}s')
