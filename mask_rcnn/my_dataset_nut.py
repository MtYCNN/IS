# Time:2022/6/2
__author__ = 'YangChen'

import os
import json
import torch
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from train_utils import coco_remove_images_without_annotations, convert_coco_poly_mask
import numpy as np
from torchvision import transforms


def get_label_map():
    return {1: 1, 2: 2, 3: 3}


transform_target = transforms.Compose(
    [transforms.ToTensor()])




class NutDetection(data.Dataset):
    def __init__(self, root, mode="train", transforms=None):
        super(NutDetection, self).__init__()
        assert mode in ["train", "val"], 'mode must be in ["train", "val"]'
        anno_file = f"trainval.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_path = os.path.join(root, f"images")
        assert os.path.exists(self.img_path), "path '{}' does not exist.".format(self.img_path)
        self.ann_path = os.path.join(root, anno_file)
        assert os.path.exists(self.ann_path), "path '{}' does not exist.".format(self.ann_path)

        self.mode = mode
        self.transforms = transforms
        self.nut = COCO(self.ann_path)

        # 获取nut数据索引与类别名称的关系
        data_classes = dict([(v["id"], v["name"]) for k, v in self.nut.cats.items()])
        max_index = max(data_classes.keys())

        # 将缺失的类别名称设置成N/A
        nut_classes = {}
        for k in range(1, max_index + 1):
            if k in data_classes:
                nut_classes[k] = data_classes[k]
            else:
                nut_classes[k] = "N/A"
        if mode == "train":
            json_str = json.dumps(nut_classes, indent=4)
            with open("nut_indices.json", "w") as f:
                f.write(json_str)

        self.nut_classes = nut_classes
        self.target_transforms = transform_target

        ids = list(sorted(self.nut.imgs.keys()))
        if mode == "train" or "val":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = coco_remove_images_without_annotations(self.nut, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self, img_id: int, coco_targets: list, w: int = None, h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        nut = self.nut
        img_id = self.ids[index]
        ann_ids = nut.getAnnIds(imgIds=img_id)
        nut_target = nut.loadAnns(ann_ids)

        path = nut.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, nut_target, w, h)

        if self.transforms is not None:
            img = self.transforms(img)
            # target = self.target_transforms(target, w, h)
        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        nut = self.nut
        img_id = self.ids[index]

        img_info = nut.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    train = NutDetection("E:/YCNN/mask_rcnn/data/", mode="train")
    print(train.get_height_and_width(0))
    print(len(train))
    t = train[0]
    print(t[1])
