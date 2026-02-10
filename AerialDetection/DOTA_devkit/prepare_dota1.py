import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO import DOTA2COCOTest, DOTA2COCOTrain
import argparse

# CODrone dataset: 12 classes
wordname_15 = ['car', 'truck', 'traffic-sign', 'people', 'motor', 'bicycle', 
               'traffic-light', 'tricycle', 'bridge', 'bus', 'boat', 'ship']

def parse_args():
    parser = argparse.ArgumentParser(description='prepare CODrone dataset')
    parser.add_argument('--srcpath', default='/path/to/CODrone')
    parser.add_argument('--dstpath', default=r'/path/to/CODrone-split-1024',
                        help='prepare data')
    args = parser.parse_args()

    return args

def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)
def filecopy(srcpath, dstpath, num_process=8):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)

def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=8):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)

def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')

def prepare(srcpath, dstpath):
    """
    :param srcpath: CODrone root path (train, val, test)
          train --> train1024, val --> val1024, test --> test1024
    :return:
    """
    # CODrone: 12 valid classes
    codrone_classes = ['car', 'truck', 'traffic-sign', 'people', 'motor', 'bicycle',
                      'traffic-light', 'tricycle', 'bridge', 'bus', 'boat', 'ship']
    
    # Create main output directory
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
        print(f"Created main output directory: {dstpath}")
    
    # Create three separate output directories
    if not os.path.exists(os.path.join(dstpath, 'test1024')):
        os.makedirs(os.path.join(dstpath, 'test1024'))
    if not os.path.exists(os.path.join(dstpath, 'train1024')):
        os.makedirs(os.path.join(dstpath, 'train1024'))
    if not os.path.exists(os.path.join(dstpath, 'val1024')):
        os.makedirs(os.path.join(dstpath, 'val1024'))

    # Split training data - output to independent train1024 directory
    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(dstpath, 'train1024'),
                      gap=512,                    # Step size: 512
                      subsize=1024,
                      num_process=8,              # Process count: 8
                      valid_classes=codrone_classes,  # Class filtering
                      ext='.jpg'                  # Image format: jpg
                      )
    split_train.splitdata(1)

    # Split validation data - output to independent val1024 directory  
    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
                       os.path.join(dstpath, 'val1024'),
                      gap=512,                    # Step size: 512
                      subsize=1024,
                      num_process=8,              # Process count: 8
                      valid_classes=codrone_classes,  # Class filtering
                      ext='.jpg'                  # Image format: jpg
                      )
    split_val.splitdata(1)

    # Split test data (if test set exists)
    if os.path.exists(os.path.join(srcpath, 'test')):
        split_test = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'test'),
                           os.path.join(dstpath, 'test1024'),
                          gap=512,                # Step size: 512
                          subsize=1024,
                          num_process=8,          # Process count: 8
                          valid_classes=codrone_classes,  # Class filtering
                          ext='.jpg'              # Image format: jpg
                          )
        split_test.splitdata(1)
        
        # Generate test set in COCO format
        DOTA2COCOTrain(os.path.join(dstpath, 'test1024'), 
                      os.path.join(dstpath, 'test1024', 'CODrone_test1024.json'), 
                       codrone_classes, difficult='-1', img_ext='.jpg')

    # Generate training set in COCO format
    DOTA2COCOTrain(os.path.join(dstpath, 'train1024'), 
                   os.path.join(dstpath, 'train1024', 'CODrone_train1024.json'), 
                   codrone_classes, difficult='-1', img_ext='.jpg')
    
    # Generate validation set in COCO format
    DOTA2COCOTrain(os.path.join(dstpath, 'val1024'), 
                   os.path.join(dstpath, 'val1024', 'CODrone_val1024.json'), 
                   codrone_classes, difficult='-1', img_ext='.jpg')

if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)