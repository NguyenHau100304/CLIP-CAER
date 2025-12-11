import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from dataloader.video_transform import *
import numpy as np
#from imblearn.over_sampling import RandomOverSampler
import cv2
from PIL import Image
from PIL import ImageDraw
import numpy as np
import json
import random

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): # 路径
        return self._data[0]

    @property       # 帧数
    def num_frames(self):
        return int(self._data[1])

    @property       # 标签
    def label(self):
        return int(self._data[2])

class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size, bounding_box_face, bounding_box_body):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = bounding_box_face
        self.bounding_box_body = bounding_box_body
        self._read_sample()
        self._parse_list()
        self._read_boxs()
        self._read_body_boxes()

    def _read_boxs(self):
        with open(self.bounding_box_face, 'r') as f:
            self.boxs = json.load(f)

    def _read_body_boxes(self):
        with open(self.bounding_box_body, 'r') as f:
            self.body_boxes = json.load(f)

    def _cv2pil(self, im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self, im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self, im, width, height):
        w, h = im.shape[1], im.shape[0]
        r = min(width / w, height / h)
        new_w, new_h = int(w * r), int(h * r)
        im = cv2.resize(im, (new_w, new_h))
        pw = (width - new_w) // 2
        ph = (height - new_h) // 2
        top, bottom = ph, ph
        left, right = pw, pw
        if top + bottom + new_h < height:
            bottom += 1
        if left + right + new_w < width:
            right += 1
        im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im, r

    def _face_detect(self, img, box, margin, mode='face'):
        if box is None:
            return img
        else:
            left, upper, right, lower = box
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)
            left = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(img.width, right + margin)
            lower = min(img.height, lower + margin)
            if mode == 'face':
                img = img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image

    def _read_sample(self):
        # Lưu ý: Python không cho phép cộng str + list trực tiếp, ta phải xử lý từng phần tử
        self.sample_list = [
            [('/kaggle/input/raer-enhanced/' if 'images_new' in x else '/kaggle/input/raer-enhanced/RAER/') + x.strip().split(' ')[0]] + x.strip().split(' ')[1:]
            for x in open(self.list_file) if x.strip()
        ]

    def _parse_list(self):
        self.video_list = [VideoRecord(item) for item in self.sample_list]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        video_frames_path = glob.glob(os.path.join(record.path, '*'))
        video_frames_path.sort()
        images = list()
        images_face = list()
        
        # --- CẤU HÌNH PREFIX ---
        # Đường dẫn gốc cần loại bỏ để khớp với Key trong JSON
        # Lưu ý: Cần có dấu '/' ở cuối để thay thế sạch sẽ
        prefix_to_remove = '/kaggle/input/raer-enhanced/' 

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                
                # Safety check: đảm bảo index không vượt quá số frame
                if p >= len(video_frames_path):
                    p = len(video_frames_path) - 1
                
                img_path = os.path.join(video_frames_path[p])
                parent_dir = os.path.dirname(img_path)
                file_name = os.path.basename(img_path)

                # --- XỬ LÝ LOOKUP KEY (QUAN TRỌNG) ---
                # Loại bỏ prefix tuyệt đối để lấy Key tương đối (vd: RAER/images/... hoặc images_new/...)
                if parent_dir.startswith(prefix_to_remove):
                    lookup_key = parent_dir.replace(prefix_to_remove, "")
                else:
                    # Trường hợp đường dẫn không bắt đầu bằng prefix (ví dụ chạy local hoặc đường dẫn khác)
                    # Ta có thể thử tìm từ vị trí xuất hiện của 'RAER/' hoặc 'images_new/'
                    if 'RAER/images' in parent_dir:
                        idx = parent_dir.find('RAER/images')
                        lookup_key = parent_dir[idx:]
                    elif 'images_new' in parent_dir:
                        idx = parent_dir.find('images_new')
                        lookup_key = parent_dir[idx:]
                    else:
                        lookup_key = parent_dir # Fallback

                # --- 1. LẤY FACE BOUNDING BOX ---
                box = None
                if lookup_key in self.boxs:
                    if file_name in self.boxs[lookup_key]:
                        box = self.boxs[lookup_key][file_name]
                        # Kiểm tra nếu box rỗng (media pipe ko bắt được) -> lấy full ảnh
                        if not box: 
                            box = None 
                    else:
                        # Có folder nhưng không có file ảnh này trong json
                        # print(f"DEBUG: [Face] Key found but file missing: {file_name} in {lookup_key}")
                        pass
                else:
                    # Không tìm thấy folder key trong json
                    print(f"DEBUG: [Face] Key not found in JSON: {lookup_key}")

                # --- 2. LẤY BODY BOUNDING BOX ---
                body_box = None
                # Body box dùng chung key với folder (lookup_key)
                if lookup_key in self.body_boxes:
                    body_box = self.body_boxes[lookup_key]
                    if not body_box: body_box = None
                else:
                    print(f"DEBUG: [Body] Key not found in JSON: {lookup_key}")

                # --- XỬ LÝ ẢNH ---
                img_pil = Image.open(img_path).convert('RGB') # Luôn convert RGB để tránh lỗi kênh màu
                img_pil_face = img_pil.copy() # Copy để crop mặt riêng

                # A. CROP BODY (Nếu có bbox thì crop, không thì lấy full)
                if body_box is not None:
                    # Đảm bảo toạ độ int
                    left, upper, right, lower = map(int, body_box)
                    img_pil_body = img_pil.crop((left, upper, right, lower))
                else:
                    img_pil_body = img_pil

                # Resize Body Image về kích thước model input (ví dụ 224x224)
                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, r = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                
                # B. CROP FACE
                # Hàm _face_detect của bạn đã xử lý logic crop, padding và trả về ảnh
                img_face_crop = self._face_detect(img_pil_face, box, margin=20, mode='face')
                
                # Resize Face Image (quan trọng: ảnh mặt sau khi crop size lộn xộn, cần resize chuẩn)
                img_cv_face = self._pil2cv(img_face_crop)
                img_cv_face, r_face = self._resize_image(img_cv_face, self.image_size, self.image_size)
                img_pil_face_final = self._cv2pil(img_cv_face)

                # Thêm vào list sequence
                seg_imgs = [img_pil_body]
                seg_imgs_face = [img_pil_face_final]

                images.extend(seg_imgs)
                images_face.extend(seg_imgs_face)
                
                # Tăng index để lấy frame tiếp theo (nếu duration > 1)
                if p < record.num_frames - 1:
                    p += 1

        # Transform (ToTensor, Normalize...)
        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        images_face = self.transform(images_face)
        images_face = torch.reshape(images_face, (-1, 3, self.image_size, self.image_size))
        
        # Trả về: (Batch Face, Batch Body, Label)
        return images_face, images, record.label - 1

    def __len__(self):
        return len(self.video_list)


def train_data_loader(list_file, num_segments, duration, image_size,dataset_name,bounding_box_face,bounding_box_body):
    if dataset_name == "RAER":
         train_transforms = torchvision.transforms.Compose([
            RandomRotation(4),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
            
    
    train_data = VideoDataset(list_file=list_file,
                              num_segments=num_segments, #16
                              duration=duration, #1
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              bounding_box_face=bounding_box_face,
                              bounding_box_body=bounding_box_body
                              )
    return train_data


def test_data_loader(list_file, num_segments, duration, image_size,bounding_box_face,bounding_box_body):
    
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    
    test_data = VideoDataset(list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             bounding_box_face=bounding_box_face,
                             bounding_box_body=bounding_box_body
                             )
    return test_data