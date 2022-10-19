import numpy as np
import pandas as pd
import cv2
import warnings
import torch
warnings.simplefilter(action='ignore', category=FutureWarning)


class PostProcess:

    def __init__(self, yolo_result, final_results,crop_size, second_conf_thresh):
        self.second_conf_thresh = second_conf_thresh
        self.crop_size = crop_size
        self.yolo_result = yolo_result
        self.final_results = final_results

    def save_image(self, path, image, name=''):
        cv2.imwrite(path + name, image)

    def split_name_from_path(self, path):
        file_name = path.split('/')[-1]
        return file_name

    def pop_row(self,df, idx, axis=0):
        row = df.iloc[idx]
        return row

    def sort_df(self,df):
        df = df.sort_values('conf_thresh', ascending=False)
        return df

    def globalize_coordinates(self):
        df = pd.read_csv(self.yolo_result, sep = ' ')
        df.columns = ['file_name','class_name','conf_thresh', 'x1', 'y1', 'x2', 'y2']
        df_new = pd.DataFrame(columns = ['file_name','class_name','conf_thresh', 'x_min', 'y_min', 'x_max', 'y_max'])
        df_new['x_min'] = df['file_name'].map(lambda x: float(x.split('_')[-1])) + df['x1']
        df_new['y_min'] = df['file_name'].map(lambda x: float(x.split('_')[-2])) + df['y1']
        df_new['x_max'] = df['file_name'].map(lambda x: float(x.split('_')[-1])) + df['x2']
        df_new['y_max'] = df['file_name'].map(lambda x: float(x.split('_')[-2])) + df['y2']
        df_new['file_name'] = df['file_name'].map(lambda x: '_'.join(x.split('_')[:-2]))
        df_new['conf_thresh'] = df['conf_thresh']
        df_new['class_name'] = df['class_name']
        return df_new

    def separate_image_results(self, df):
        g = df.groupby('file_name')
        return g

    def plot(self, img, name, image_bboxes):
        colors = {'OST':(255,0,255),'human': (255,0,0),'sup-board':(0,255,0),'boat':(0,0,255),'bouy': (255,255,0) ,'sailboat':(0,255,255), 'kayak': (0,127,255)}
        for idx in range(len(image_bboxes)):
            row = image_bboxes.iloc[idx,:]
            conf = row.conf_thresh
            x1 = int(row.x_min)
            x2 = int(row.x_max)
            y1 = int(row.y_min)
            y2 = int(row.y_max)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imwrite(self.final_results + name + '.jpg', img)

    def iou(self, box1, box2):
        box1 = torch.tensor(box1)
        box2 = torch.tensor(box2)
        box1_x1 = box1[0]
        box1_y1 = box1[1]
        box1_x2 = box1[2]
        box1_y2 = box1[3]
        box2_x1 = box2[0]
        box2_y1 = box2[1]
        box2_x2 = box2[2]
        box2_y2 = box2[3]
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
        return intersection / (box1_area + box2_area - intersection + 1e-6)

    def nms(self,df):
        df = self.sort_df(df)
        df['id'] = np.arange(0, len(df))
        df.reset_index(drop=True)
        df.set_index('id', inplace=True)
        df_result = pd.DataFrame(columns=pd.DataFrame(
            columns=['file_name', 'class_name', 'conf_thresh', 'x_min', 'y_min', 'x_max', 'y_max']))
        while len(df) != 0:
            row1 = self.pop_row(df, [0])
            df = df.drop([0])
            df['id'] = np.arange(0, len(df))
            df.set_index('id', inplace=True)
            idx_list = []
            for idx in range(len(df)):

                row2 = df.iloc[idx]
                x11 = int(row1.x_min)
                x21 = int(row1.x_max)
                y11 = int(row1.y_min)
                y21 = int(row1.y_max)
                x12 = int(row2.x_min)
                x22 = int(row2.x_max)
                y12 = int(row2.y_min)
                y22 = int(row2.y_max)

                bbox1 = [x11, y11, x21, y21]
                bbox2 = [x12, y12, x22, y22]

                row1_new = [row1.file_name, row1.class_name, row1.conf_thresh, x11, y11, x21, y21]
                iou_score = self.iou(bbox1, bbox2)
                if iou_score >= self.second_conf_thresh:
                    idx_list.append(idx)
            df.drop(idx_list, axis=0, inplace=True)
            df['id'] = np.arange(0, len(df))
            df.set_index('id', inplace=True)
            df_result = df_result.append(pd.Series(row1_new,
                                                   index=['file_name', 'class_name', 'conf_thresh', 'x_min', 'y_min',
                                                          'x_max', 'y_max']), ignore_index=True)
        return df_result
