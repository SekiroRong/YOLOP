#-------------------------------------------------------------------------------------------------#
#   kmeans虽然会对数据集中的框进行聚类，但是很多数据集由于框的大小相近，聚类出来的9个框相差不大，
#   这样的框反而不利于模型的训练。因为不同的特征层适合不同大小的先验框，越浅的特征层适合越大的先验框
#   原始网络的先验框已经按大中小比例分配好了，不进行聚类也会有非常好的效果。
#-------------------------------------------------------------------------------------------------#
import glob
import xml.etree.ElementTree as ET

import numpy as np

def bbox3Dtowh(bboxs,data):
    bbox2d = []
    for bbox in bboxs:
        bbox_class = bbox[0]
        bbox3d = bbox[1]
        x, y, x2, y2 = bbox3d[8][0], bbox3d[8][1], bbox3d[9][0], bbox3d[9][1]
        if x == y == x2 == y2 == 0:
            continue
        # b2d = [min(bbox3d[0][0], bbox3d[1][0], bbox3d[2][0], bbox3d[3][0]),
        #        min(bbox3d[4][1], bbox3d[5][1], bbox3d[6][1], bbox3d[7][1]),
        #        max(bbox3d[4][0], bbox3d[5][0], bbox3d[6][0], bbox3d[7][0]),
        #        max(bbox3d[0][1], bbox3d[1][1], bbox3d[2][1], bbox3d[4][1])]
        data.append([x2-x, y2-y])

    # print(bbox2d)
    # return np.array(bbox2d)


def parse3Dbbox(path,data):
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
    bbox3Dtowh(labels,data)

def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])

def kmeans(box,k):
    #-------------------------------------------------------------#
    #   取出一共有多少框
    #-------------------------------------------------------------#
    row = box.shape[0]
    
    #-------------------------------------------------------------#
    #   每个框各个点的位置
    #-------------------------------------------------------------#
    distance = np.empty((row,k))
    
    #-------------------------------------------------------------#
    #   最后的聚类位置
    #-------------------------------------------------------------#
    last_clu = np.zeros((row,))

    np.random.seed()

    #-------------------------------------------------------------#
    #   随机选5个当聚类中心
    #-------------------------------------------------------------#
    cluster = box[np.random.choice(row,k,replace = False)]
    while True:
        #-------------------------------------------------------------#
        #   计算每一行距离五个点的iou情况。
        #-------------------------------------------------------------#
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        #-------------------------------------------------------------#
        #   取出最小点
        #-------------------------------------------------------------#
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        #-------------------------------------------------------------#
        #   求每一个类的中位点
        #-------------------------------------------------------------#
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []

    for filename in (glob.glob(path + '/*.txt')):
        parse3Dbbox(filename,data)
    #-------------------------------------------------------------#
    #   对于每一个xml都寻找box
    #-------------------------------------------------------------#
    # for xml_file in glob.glob('{}/*xml'.format(path)):
    #     tree = ET.parse(xml_file)
    #     height = int(tree.findtext('./size/height'))
    #     width = int(tree.findtext('./size/width'))
    #     if height<=0 or width<=0:
    #         continue
    #
    #     #-------------------------------------------------------------#
    #     #   对于每一个目标都获得它的宽高
    #     #-------------------------------------------------------------#
    #     for obj in tree.iter('object'):
    #         xmin = int(float(obj.findtext('bndbox/xmin'))) / width
    #         ymin = int(float(obj.findtext('bndbox/ymin'))) / height
    #         xmax = int(float(obj.findtext('bndbox/xmax'))) / width
    #         ymax = int(float(obj.findtext('bndbox/ymax'))) / height
    #
    #         xmin = np.float64(xmin)
    #         ymin = np.float64(ymin)
    #         xmax = np.float64(xmax)
    #         ymax = np.float64(ymax)
    #         # 得到宽高
    #         data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    #-------------------------------------------------------------#
    #   运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    #   会生成yolo_anchors.txt
    #-------------------------------------------------------------#
    SIZE_x        = 640
    SIZE_y = 480
    anchors_num = 9
    #-------------------------------------------------------------#
    #   载入数据集，可以使用VOC的xml
    #-------------------------------------------------------------#
    path        = r'G:\Carla_Dataset\3Dbbox\train'
    
    #-------------------------------------------------------------#
    #   载入所有的xml
    #   存储格式为转化为比例后的width,height
    #-------------------------------------------------------------#
    data = load_data(path)
    
    #-------------------------------------------------------------#
    #   使用k聚类算法
    #-------------------------------------------------------------#
    out = kmeans(data,anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE_x)
    data = out
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()
