from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input,cvtColor2gray
from config import depth_img

rgb_path = r'G:\Carla_Recorder\Cam_Recorder'
depth_path = r'G:\Carla_Recorder\Depth_Recorder'

dataset_3dbbox_path = r"G:\Carla_Dataset\3Dbbox"
dataset_laneline_path = r"G:\Carla_Dataset\LaneLine"
dataset_drivearea_path = r"G:\Carla_Dataset\DriveableArea"
dataset_img_path = r"G:\Carla_Dataset\Image"


def bbox3Dto2Dxywh(bboxs):
    bbox2d = []
    for bbox in bboxs:
        bbox_class = bbox[0]
        bbox3d = bbox[1]
        # b2d = [min(bbox3d[0][0], bbox3d[1][0], bbox3d[2][0], bbox3d[3][0]),
        #        min(bbox3d[4][1], bbox3d[5][1], bbox3d[6][1], bbox3d[7][1]),
        #        max(bbox3d[4][0], bbox3d[5][0], bbox3d[6][0], bbox3d[7][0]),
        #        max(bbox3d[0][1], bbox3d[1][1], bbox3d[2][1], bbox3d[4][1])]
        # # w,h = (b2d[2] - b2d[0]), (b2d[3] - b2d[1])
        # x,y,x2,y2 = b2d[0], b2d[1],b2d[2], b2d[3]
        # print(x,y,w,h)
        x, y, x2, y2 = bbox3d[8][0],bbox3d[8][1],bbox3d[9][0],bbox3d[9][1]
        if x == y == x2 == y2 == 0:
            continue
        if bbox_class == 'Car':
            bbox2d.append([x,y,x2,y2, 1,0])
        elif bbox_class == 'Pedestrian':
            bbox2d.append([x,y,x2,y2, 0,1])
        # bbox2d.append([b2d[0], b2d[1], b2d[2] - b2d[0], b2d[3] - b2d[1]])

    # print(bbox2d)
    return bbox2d


def parse3Dbbox(path):
    with open(path, 'r') as f:
        labels = []
        label = []
        point = []
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            label.append(line[0])
            for i in range(int(len(line) / 2) - 1):
                point.append((int(line[2 * i + 1]), int(line[2 * i + 2])))
            label.append(point)
            point = []
            labels.append(label)
            label = []

    # print(labels)
    return bbox3Dto2Dxywh(labels)


def parseLaneline(path):
    with open(path, 'r') as f:
        labels = []
        label = []
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            for i in range(int(len(line) / 2)):
                label.append((int(line[2 * i]), int(line[2 * i + 1])))
            labels.append(label)
            label = []

    # print(labels)
    return labels


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, mosaic, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.length = len(self.annotation_lines)
        self.mosaic = mosaic
        self.train = train

        if train:
            status = 'train'
        else:
            status = 'val'

        # datatxt_full = annotation_lines + '/' + status + '\images.txt'

        imgs = []
        img_path = open(annotation_lines, 'r')  # 打开txt，读取内容
        for line in img_path:  # 按行循环txt文本中的内容
            line = line.strip('\n')  # 删除本行string字符串末尾的指定字符
            path = line.strip(rgb_path)
            path = path.strip(r'/train/')
            path = path.strip(r'/val/')
            path = path.strip('.jpg')
            bbox3d_path = dataset_3dbbox_path + '/' + status + path + '.txt'
            laneline_path = dataset_laneline_path + '/' + status + path + '.jpg'
            drivearea_path = dataset_drivearea_path + '/' + status + path + '.jpg'

            bbox2d = parse3Dbbox(bbox3d_path)
            # ll_path = parseLaneline(laneline_path)

            # bbox2d = bbox3Dto2Dxywh(bbox3d)
            # print((line, bbox2d, laneline_path, drivearea_path))

            imgs.append((line, bbox2d, laneline_path, drivearea_path))

        self.imgs = imgs

    def __len__(self):  # 它返回的是数据集的长度，必须有
        return len(self.imgs)

    def __getitem__(self, index):
        index = index % self.length

        # img_path, bbox2d, ll, drivearea_path = self.imgs[index]
        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#

        # mosaic暂时停用
        # if self.mosaic:
        #     if self.rand() < 0.5:
        #         lines = sample(self.annotation_lines, 3)
        #         lines.append(self.annotation_lines[index])
        #         shuffle(lines)
        #         image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
        #     else:
        #         image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        # else:
        #     image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        if depth_img:
            image, box, ll_data, da_data, Dimage = self.get_random_data(self.imgs[index], self.input_shape, random=self.train)
            Dimage = np.transpose(preprocess_input(np.array(Dimage, dtype=np.float32)), (2, 0, 1))
        else:
            image, box, ll_data, da_data = self.get_random_data(self.imgs[index], self.input_shape, random=self.train)

        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        ll_data = np.transpose(preprocess_input(np.array(ll_data, dtype=np.float32)), (2, 0, 1))
        da_data = np.transpose(preprocess_input(np.array(da_data, dtype=np.float32)), (2, 0, 1))

        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

        if depth_img:
            return image, box, ll_data, da_data, Dimage
        return image, box, ll_data, da_data

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, imgs, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        # line = annotation_line.split()
        # print(imgs.shape)
        img_path, bbox2d, laneline_path, drivearea_path = imgs
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(img_path)
        image = cvtColor(image)

        if depth_img:
            path = img_path.strip(rgb_path)
            dp = depth_path + path
            Dimage = Image.open(dp)
            Dimage = cvtColor(Dimage)

        ll = Image.open(laneline_path)

        ll = cvtColor(ll)
        da = Image.open(drivearea_path)
        da = cvtColor(da)

        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box))) for box in bbox2d])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            ll = ll.resize((nw, nh), Image.BICUBIC)
            da = da.resize((nw, nh), Image.BICUBIC)
            if depth_img:
                Dimage = Dimage.resize((nw, nh), Image.BICUBIC)
                Dnew_image = Image.new('RGB', (w, h), (0, 0, 0))
                Dnew_image.paste(Dimage, (dx, dy))
                Dimage_data = np.array(Dnew_image, np.float32)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_ll = Image.new('RGB', (w, h), (0, 0, 0))
            new_da = Image.new('RGB', (w, h), (0, 0, 0))
            new_image.paste(image, (dx, dy))
            new_ll.paste(ll, (dx, dy))
            new_da.paste(da, (dx, dy))
            image_data = np.array(new_image, np.float32)
            ll_data = np.array(new_ll, np.uint8)
            da_data = np.array(new_da, np.uint8)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            if depth_img:
                return image_data, box, ll_data, da_data, Dimage_data

            return image_data, box, ll_data, da_data

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        ll = ll.resize((nw, nh), Image.BICUBIC)
        da = da.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_ll = Image.new('RGB', (w, h), (0, 0, 0))
        new_da = Image.new('RGB', (w, h), (0, 0, 0))
        new_image.paste(image, (dx, dy))
        new_ll.paste(ll, (dx, dy))
        new_da.paste(da, (dx, dy))
        image = new_image
        ll = new_ll
        da = new_da

        if depth_img:
            Dimage = Dimage.resize((nw, nh), Image.BICUBIC)
            Dnew_image = Image.new('RGB', (w, h), (0, 0, 0))
            Dnew_image.paste(Dimage, (dx, dy))
            Dimage = Dnew_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            ll = ll.transpose(Image.FLIP_LEFT_RIGHT)
            da = da.transpose(Image.FLIP_LEFT_RIGHT)
            if depth_img:
                Dimage = Dimage.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   色域扭曲
        # ------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        if depth_img:
            return image_data, box, ll, da, Dimage
        return image_data, box, ll, da

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, max_boxes=100, hue=.1, sat=1.5, val=1.5):
        h, w = input_shape
        min_offset_x = self.rand(0.25, 0.75)
        min_offset_y = self.rand(0.25, 0.75)

        nws = [int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)),
               int(w * self.rand(0.4, 1))]
        nhs = [int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)),
               int(h * self.rand(0.4, 1))]

        place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1], int(w * min_offset_x),
                   int(w * min_offset_x)]
        place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y), int(h * min_offset_y),
                   int(h * min_offset_y) - nhs[3]]

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            nw = nws[index]
            nh = nhs[index]
            image = image.resize((nw, nh), Image.BICUBIC)

            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 进行色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(new_image / 255, np.float32), cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # 对框进行进一步的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    lls = []
    das = []
    dis = []
    if depth_img:
        for img, box, ll, da, di in batch:
            images.append(img)
            bboxes.append(box)
            lls.append(ll)
            das.append(da)
            dis.append(di)
        images = np.array(images)
        lls = np.array(lls)
        das = np.array(das)
        dis = np.array(dis)
        return images, bboxes, lls, das, dis
    else:
        for img, box, ll, da in batch:
            images.append(img)
            bboxes.append(box)
            lls.append(ll)
            das.append(da)
        images = np.array(images)
        lls = np.array(lls)
        das = np.array(das)
        return images, bboxes, lls, das
