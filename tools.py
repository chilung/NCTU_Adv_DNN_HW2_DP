import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def read_json(path):
    '''
    path: the json file path
    return:
        im_name: the name of the images 
        bboxes: the boxes of the table in the table, the loc_cord is scaled to [0,1]
        min_y/h, max_y/h: the table location, this is used to remove the header line and footer line.
    '''
    # print(path)
    with open(path, 'r') as f:
        json_dict = json.load(f)
    im_name = json_dict["imagePath"]
    h = json_dict['imageHeight']
    w = json_dict['imageWidth']
    bboxes = []
    min_y = 10000
    max_y = 0
    for shape in json_dict['shapes']:
        # label = int(shape["label"])
        # if label != 1:
        #     print(f'{path} label is {label}!!!!!!!!!!!!!')
            
        if len(shape["points"]) == 2:
            min_xy, max_xy = shape["points"]
            x1 = min([min_xy[0], max_xy[0]])
            y1 = min([min_xy[1], max_xy[1]])
            x2 = max([min_xy[0], max_xy[0]])
            y2 = max([min_xy[1], max_xy[1]])
        elif len(shape["points"]) == 4:
            x1y1, x2y2, x3y3, x4y4 = shape["points"]
            x1 = min([x1y1[0], x2y2[0], x3y3[0], x4y4[0]])
            y1 = min([x1y1[1], x2y2[1], x3y3[1], x4y4[1]])
            x2 = max([x1y1[0], x2y2[0], x3y3[0], x4y4[0]])
            y2 = max([x1y1[1], x2y2[1], x3y3[1], x4y4[1]])
        
        if x1 < 0:
            x1 = 0. 
        if y1 < 0:
            y1 = 0.
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h
        if x1>x2 or y1>y2:
            print(f'{path} {x1} {y1} {x2} {y2}')
        bboxes.append([x1, y1, x2, y2])
        if y1 < min_y:
            min_y = y1
        if y2 > max_y:
            max_y = y2

    return im_name, bboxes, min_y, max_y

def json2HVmask(path, image, width, height, p_crack=0):
    '''
    path: the path the segment json file
    image: the origin image: RGB
    return:
        mask: np.array, uint8, (h, w), max_value=1
    '''
    im = image.copy()
    thickness = 5
    if height < 1000:
        thickness = 3
    if height > 2000:
        thickness = 8

    with open(path, 'r') as f:
        json_dict = json.load(f)
    im_name = json_dict["imagePath"]
    h = json_dict['imageHeight']
    w = json_dict['imageWidth']
    Hmask = np.zeros((height, width), np.uint8)
    Vmask = np.zeros((height, width), np.uint8)
    for shape in json_dict['shapes']:
        if len(shape["points"]) == 2:
            min_xy, max_xy = shape["points"]
            x1 = min([min_xy[0], max_xy[0]])
            y1 = min([min_xy[1], max_xy[1]])
            x2 = max([min_xy[0], max_xy[0]])
            y2 = max([min_xy[1], max_xy[1]])
        else:
            print('Error in json2mask, the points arenot 2.')
            sys.exit()
        
        if x1 < 0:
            x1 = 0. 
        if y1 < 0:
            y1 = 0.
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h

        # crack line
        if random.random() < 0.5: # crack line prob=0.5
            if x2 - x1 < 5: # vline
                two_num = random.sample(range(int(y1), int(y2)), 2)
                min_num = min(two_num)
                max_num = max(two_num)
                x = int((x1+x2)/2)
                im[min_num: max_num, (x-1):(x+1)] = im[min_num: max_num, (x+2):(x+4)]
            elif y2 - y1 < 5: # hlie
                two_num = random.sample(range(int(x1), int(x2)), 2)
                min_num = min(two_num)
                max_num = max(two_num)
                y = int((y1+y2)/2)
                im[(y-1):(y+1), min_num: max_num] = im[(y+2):(y+4), min_num: max_num]

        # mask
        a = y2 - y1
        b = x2 - x1
        c = (a**2 + b**2)**0.5
        sin = (a + 1e-6) / (c + 1e-6)
        x1, x2 = x1 / w, x2 / w
        y1, y2 = y1 / h, y2 / h
        # Horizontal line
        if abs(sin) < 0.2:     
            cv2.line(Hmask, (int(x1*width), int(y1*height)), (int(x2*width), int(y2*height)), (255), thickness=thickness)
        # Vertical line
        if abs(sin) > 0.8:
            cv2.line(Vmask, (int(x1*width), int(y1*height)), (int(x2*width), int(y2*height)), (255), thickness=thickness)

        

    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, Hmask = cv2.threshold(Hmask, 127, 1, cv2.THRESH_BINARY)
    ret, Vmask = cv2.threshold(Vmask, 127, 1, cv2.THRESH_BINARY)

    if random.random() > p_crack: # every image has the `p_crack` prob to do the augs
        im = image

    return Hmask, Vmask, im

def prepare_data():
    for mode in ['train', 'val']:
        writer = open(f'./data/{mode}.txt', 'w')
        with open(f'../Table-Detection-and-Structure-Recognition/DeepLabV3+/data/{mode}.txt', 'r') as f:
            for line in tqdm(f.readlines()):
                im_path = line.strip()
                det_path = im_path.replace('image', 'detect').replace('png', 'json')
                im_name, bboxes, _, _ = read_json(det_path)
                relative_path = im_path[15:]
                im_info = ''
                im_info += relative_path
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    im_info += f' {x1},{y1},{x2},{y2},0'
                im_info += '\n'
                writer.write(im_info)
        writer.close()

if __name__ == '__main__':
    prepare_data()

