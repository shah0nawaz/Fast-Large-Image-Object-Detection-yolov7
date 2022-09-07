import concurrent.futures
import os
import cv2
import glob
import time

class CropFast:
    def __init__(self, input_path ,crops_path, crop_size, step_size):
        self.input_path = input_path
        self.output = crops_path
        self.crop_size = crop_size
        self.step_size = step_size

    def slicing_patches(self):
        image_paths = glob.glob(self.input_path + '*')
        print(f'Cropping Process In Progress ...')
        tc = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.crop_thread, image_paths)
        print(f'Cropping Process finished in {time.time()-tc} Seconds')

    def crop_thread(self,image_path):
        points = self.divisions(image_path)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.cropping, points)
        return image_path, points, results

    def split_name_from_path(self, path):
        file_name = path.split('/')[-1]
        return file_name

    def divisions(self,image_path):
        self.image_array = {}
        name_ext = self.split_name_from_path(image_path)
        name = name_ext.split('.')[0]
        image = cv2.imread(image_path)
        self.image_array[name] = image
        (height, width, channels) = image.shape
        w1 = self.step_size
        h1 = self.step_size

        points = []
        for y in range(0, height-(height%h1) + 1 , h1):
            for x in range(0, width-(width%w1)+ 1 , w1):

                if x + self.crop_size > width:
                    diff = width - x
                    add_value = self.crop_size - diff
                    x = x - add_value
                    start_x = x
                    start_y = y
                    point = (start_x, start_y, name)
                    points.append(point)

                elif y + self.crop_size > height:
                    diff = height - y
                    add_value = self.crop_size - diff
                    y = y - add_value
                    start_x = x
                    start_y = y
                    point = (start_x, start_y, name)
                    points.append(point)

                elif (x + self.crop_size > width) and ( y + self.crop_size > height) :
                    diff = width - x
                    add_value = self.crop_size - diff
                    x = x - add_value

                    diff = height - y
                    add_value = self.crop_size - diff
                    y = y - add_value

                    start_x = x
                    start_y = y
                    point = (start_x, start_y, name)
                    points.append(point)

                else:
                    start_x = x
                    start_y = y
                    point = (start_x, start_y, name)
                    points.append(point)
                    # break
        return points

    def cropping(self, point):
        (x1, y1, name) = point
        w = self.crop_size
        h = self.crop_size
        image_patch = self.image_array[name][y1:y1 + h, x1:x1 + w]
        cv2.imwrite(self.output + name + '_' + str(y1) + '_' + str(x1) + '.jpg', image_patch)
        return point
