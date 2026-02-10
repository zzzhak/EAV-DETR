import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from src.zoo.eavdetr.oriented_ops import box_cxcywh_to_xyxyxyxy


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()

        if bboxes.shape[-1] == 5:
            if bboxes.numel() > 0:
                # It's an oriented box (cx, cy, w, h, a), normalized.
                # Convert to an un-normalized, axis-aligned XYXY box for COCO eval.
                h, w = img.shape[-2], img.shape[-1]
                
                # De-normalize from [0,1] to pixel coordinates
                bboxes[:, [0, 2]] *= w
                bboxes[:, [1, 3]] *= h
                
                # Convert oriented cx,cy,w,h,a to 8-point polygon
                polygons = box_cxcywh_to_xyxyxyxy(bboxes)
                
                # Calculate axis-aligned bounding box (AABB) from polygon
                x_coords = polygons[:, 0::2]
                y_coords = polygons[:, 1::2]
                x_min = torch.min(x_coords, dim=1).values
                y_min = torch.min(y_coords, dim=1).values
                x_max = torch.max(x_coords, dim=1).values
                y_max = torch.max(y_coords, dim=1).values
                bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
            else:
                # Handle empty tensor case, ensure it has 4 columns
                bboxes = torch.zeros((0, 4), device=bboxes.device, dtype=bboxes.dtype)

        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    # FIXME: This is... awful?
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    
    # Handle CODrone and other non-COCO datasets
    if hasattr(dataset, 'CLASSES'):
        # Create a minimal COCO-compatible structure for evaluation
        from pycocotools.coco import COCO
        coco_gt = COCO()
        
        # Create images list based on dataset size
        images = []
        for i in range(len(dataset)):
            images.append({
                'id': i,
                'height': 1024,  # Default height for CODrone
                'width': 1024,   # Default width for CODrone  
                'file_name': f'image_{i}.jpg'
            })
        
        # Create minimal dataset structure
        coco_gt.dataset = {
            'info': {'description': f'{dataset.__class__.__name__} Dataset'},
            'images': images,
            'annotations': [],
            'categories': [{'id': i, 'name': name} for i, name in enumerate(dataset.CLASSES)]
        }
        
        # Create category mapping (0-based indexing for CODrone)
        coco_gt.cats = {i: {'id': i, 'name': name} for i, name in enumerate(dataset.CLASSES)}
        coco_gt.imgToAnns = {i: [] for i in range(len(dataset))}
        coco_gt.catToImgs = {i: [] for i in range(len(dataset.CLASSES))}
        
        # Create index for COCO
        coco_gt.createIndex()
        
        return coco_gt
        
    return convert_to_coco_api(dataset)


