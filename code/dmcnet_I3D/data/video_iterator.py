import os
import cv2
import numpy as np

import torch.utils.data as data
import logging
from PIL import Image
import functools


def pil_loader(path, modality = 'rgb', path_y = None):
    if modality == 'rgb':
        img = Image.open(path)
        rgb_img = img.convert('RGB')
    elif modality == 'flow':
        img = Image.open(path)
        img = img.convert('L')
        img_x = np.asarray(img)
        img = Image.open(path)
        img = img.convert('L')
        img_y = np.asarray(img)
        rgb_img = np.concatenate((img_x.reshape((img_x.shape[0], img_x.shape[1], 1)), img_y.reshape((img_x.shape[0], img_x.shape[1], 1))), axis = 2)
    return rgb_img


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality = 'rgb', image_loader = pil_loader):
    video = []
    video_flow= []
    for i in frame_indices:
        if modality == 'rgb':
            image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality = modality))
        elif modality == 'flow':
            image_path = os.path.join(video_dir_path, 'flow_x_{:05d}.jpg'.format(i))   
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality, os.path.join(video_dir_path, 'flow_y_{:05d}.jpg'.format(i))))
        elif modality == 'rgb+flow':
            image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, 'rgb'))
            image_path = os.path.join(video_dir_path, 'flow_x_{:05d}.jpg'.format(i))   
            if os.path.exists(image_path):
                video_flow.append(image_loader(image_path, 'flow', os.path.join(video_dir_path, 'flow_y_{:05d}.jpg'.format(i))))
    if modality == 'rgb+flow':
        return video, video_flow
    else:
        return video

GOP_SIZE = 12
#_mv_minmaxnorm = 0
def clip_and_scale(img, bound):
    img = img.astype(np.float)
    img *= 127.5 / bound
    return img

def get_gop_pos(frame_idx, representation):
    """given frame idx to find the group idx and the position inside a group"""
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv', 'flow']:
        if gop_pos == 0:    # use the next frame's residual and MV if it's iframe
            gop_pos = 1
    else:
        gop_pos = 0 # indeed find the iframe rather than rgb frame in the middle of GOP
    return gop_index, gop_pos

def mploader(video_prefix, vid_subpath, sampled_idxs, modality, accumulate, ds_factor, _mv_minmaxnorm, mv_loadimg):

    if not mv_loadimg:
        from coviar import get_num_frames
        from coviar import load
    image_loader = get_default_image_loader()
    mv = []
    res = []
    index_pos = []
    video = []
    # change this part according your relative path between frame image and videos and stored MV and R
    if video_prefix == '/HMDB51/fb/TSN_input/':
        video_path = os.path.join('/HMDB51/fb/videos_mpeg4/', vid_subpath + '.mp4')
        if not mv_loadimg:
            video_prefix2 = '/'
        else:
            video_prefix2 = '/HMDB51/fb/MC_input/'
    elif video_prefix == '/UCF101/TSN_input/':
        video_path = os.path.join('/UCF101/fb/mpeg4_videos/', vid_subpath + '.mp4')
        if not mv_loadimg:
            video_prefix2 = '/'
        else:
            video_prefix2 = '/UCF101_MC_input/'
    for frame_idx in sampled_idxs:
        index_pos.append(get_gop_pos(frame_idx, 'mv'))
    if modality == 'mv':        
        for j, item in enumerate(index_pos):
            image_path = os.path.join(video_prefix2, vid_subpath, 'mv_x_{:05d}.jpg'.format(sampled_idxs[j] + 1))
            if mv_loadimg and os.path.exists(image_path):
                img = np.asarray(image_loader(image_path, 'flow', os.path.join(video_prefix2, vid_subpath, 'mv_y_{:05d}.jpg'.format(sampled_idxs[j] +1)))).astype(np.float)
                img += -128
            else:
                gop_index, gop_pos = item
                img = load(video_path, gop_index, gop_pos, 1, accumulate)
            if _mv_minmaxnorm == 1:
                img = clip_and_scale(img, 20)   # scale values from +-20 to +-127.5
            img += 128
            img = (np.minimum(np.maximum(img, 0), 255))
            mv.append(img.astype(np.uint8))
        return mv
    elif modality == 'res':       
        for j, item in enumerate(index_pos):
            image_path = os.path.join(video_prefix2, vid_subpath, 'res_{:05d}.jpg'.format(sampled_idxs[j] + 1))
            if mv_loadimg and os.path.exists(image_path):
                img = np.asarray(image_loader(image_path, 'rgb'))
            else:
                gop_index, gop_pos = item
                img = load(video_path, gop_index, gop_pos, 2, accumulate)
                img += 128
                img = (np.minimum(np.maximum(img, 0), 255))
            res.append(img.astype(np.uint8))
        return res
    elif modality == 'I':
        for item in index_pos:
            gop_index, gop_pos = item
            img = load(video_path, gop_index, 0, 0, accumulate)
            img = img[..., ::-1]
            res.append(img.astype(np.uint8))
        return res
    elif modality == 'flow+mp4':
        
        for i in range(len(sampled_idxs)):
            image_path = os.path.join(video_prefix, vid_subpath, 'flow_x_{:05d}.jpg'.format(sampled_idxs[i] + 1))   
            if os.path.exists(image_path):
                 flow = image_loader(image_path, 'flow', os.path.join(video_prefix, vid_subpath, 'flow_y_{:05d}.jpg'.format(sampled_idxs[i] + 1)))
            image_path = os.path.join(video_prefix2, vid_subpath, 'mv_x_{:05d}.jpg'.format(sampled_idxs[i] + 1))
            if mv_loadimg and os.path.exists(image_path):
                img = image_loader(image_path, 'flow', os.path.join(video_prefix2, vid_subpath, 'mv_y_{:05d}.jpg'.format(sampled_idxs[i] +1))).astype(np.float)
                img += -128
            else:
                gop_index, gop_pos = index_pos[i]
                img = load(video_path, gop_index, gop_pos, 1, accumulate)
            if _mv_minmaxnorm == 1:
                img = clip_and_scale(img, 20)   # scale values from +-20 to +-127.5
            img += 128
            img = (np.minimum(np.maximum(img, 0), 255))
            flow = np.concatenate((flow,img), axis = 2)
            image_path = os.path.join(video_prefix2, vid_subpath, 'res_{:05d}.jpg'.format(sampled_idxs[i] + 1))
            if mv_loadimg and os.path.exists(image_path):
                img = image_loader(image_path, 'rgb')
            else:
                gop_index, gop_pos = index_pos[i]
                img = load(video_path, gop_index, gop_pos, 2, accumulate)
                img += 128
                img = (np.minimum(np.maximum(img, 0), 255))
            flow = np.concatenate((flow,img), axis = 2)
            video.append(flow.astype(np.uint8))
        return video
    

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

class Video(object):
    """basic Video class"""

    def __init__(self, vid_path):
        self.open(vid_path)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def reset(self):
        self.close()
        self.vid_path = None
        self.frame_count = -1
        self.faulty_frame = None
        return self

    def open(self, vid_path):
        assert os.path.exists(vid_path), "VideoIter:: cannot locate: `{}'".format(vid_path)

        # close previous video & reset variables
        self.reset()

        # try to open video
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            self.cap = cap
            self.vid_path = vid_path
        else:
            raise IOError("VideoIter:: failed to open video: `{}'".format(vid_path))

        return self

    def count_frames(self, check_validity=False):
        offset = 0
        if self.vid_path.endswith('.flv'):
            offset = -1
        unverified_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + offset
        if check_validity:
            verified_frame_count = 0
            for i in range(unverified_frame_count):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                if not self.cap.grab():
                    logging.warning("VideoIter:: >> frame (start from 0) {} corrupted in {}".format(i, self.vid_path))
                    break
                verified_frame_count = i + 1
            self.frame_count = verified_frame_count
        else:
            self.frame_count = unverified_frame_count
        assert self.frame_count > 0, "VideoIter:: Video: `{}' has no frames".format(self.vid_path)
        return self.frame_count

    def extract_frames(self, idxs, force_color=True):
        frames = self.extract_frames_fast(idxs, force_color)
        if frames is None:
            # try slow method:
            frames = self.extract_frames_slow(idxs, force_color)
        return frames

    def extract_frames_fast(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = []
        pre_idx = max(idxs)
        for idx in idxs:
            assert (self.frame_count < 0) or (idx < self.frame_count), \
                "idxs: {} > total valid frames({})".format(idxs, self.frame_count)
            if pre_idx != (idx - 1):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read() # in BGR/GRAY format
            pre_idx = idx
            if not res:
                self.faulty_frame = idx
                return None
            if len(frame.shape) < 3:
                if force_color:
                    # Convert Gray to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames

    def extract_frames_slow(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = [None] * len(idxs)
        idx = min(idxs)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        while idx <= max(idxs):
            res, frame = self.cap.read() # in BGR/GRAY format
            if not res:
                # end of the video
                self.faulty_frame = idx
                return None
            if idx in idxs:
                # fond a frame
                if len(frame.shape) < 3:
                    if force_color:
                        # Convert Gray to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pos = [k for k, i in enumerate(idxs) if i == idx]
                for k in pos:
                    frames[k] = frame
            idx += 1
        return frames

    def close(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        return self


class VideoIter(data.Dataset):

    def __init__(self,
                 video_prefix,
                 txt_list,
                 sampler,
                 video_transform=None,
                 name="<NO_NAME>",
                 force_color=True,
                 cached_info_path=None,
                 return_item_subpath=False,
                 shuffle_list_seed=None,
                 check_video=False,
                 tolerant_corrupted_video=None,
                 load_from_img=False,
                 modality = 'rgb',
                 accumulate = True,
                 mv_minmaxnorm = False,
                 mv_loadimg = False,
                 transform_flow = None,
                 ds_factor =16):
        super(VideoIter, self).__init__()
        # load params
        self.sampler = sampler
        self.force_color = force_color
        self.video_prefix = video_prefix
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath
        self.backup_item = None
        self.load_from_img = load_from_img
        self.modality = modality
        self.accumulate = accumulate
        self.ds_factor = ds_factor
        self.transform_flow = transform_flow
        self.mv_minmaxnorm = mv_minmaxnorm
        self.mv_loadimg =mv_loadimg 
        if (not check_video) and (tolerant_corrupted_video is None):
            logging.warning("VideoIter:: >> `check_video' is off, `tolerant_corrupted_video' is automatically activated.")
            tolerant_corrupted_video = True
        self.tolerant_corrupted_video = tolerant_corrupted_video
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(video_prefix=video_prefix,
                                               txt_list=txt_list,
                                               check_video=check_video,
                                               cached_info_path=cached_info_path)
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.video_list)
        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    def getitem_from_raw_video(self, index):
        # get current video info
        v_id, label, vid_subpath, frame_count = self.video_list[index]
        video_path = os.path.join(self.video_prefix, vid_subpath)

        faulty_frames = []
        successfule_trial = False
        try:
            if self.load_from_img:
                #print(frame_count)

                for i_trial in range(20):
                    # dynamic sampling
                    sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=v_id, prev_failed=(i_trial>0))
                    if set(list(sampled_idxs)).intersection(faulty_frames):
                        continue
                    prev_sampled_idxs = list(sampled_idxs)
                    # extracting frames
                    loader = get_default_video_loader()
                    image_indices = np.asarray(range(1, frame_count + 1))
                    if self.modality == 'rgb+flow':
                        sampled_frames, flow = loader(video_path, (image_indices[sampled_idxs]).tolist(), self.modality)
                    elif self.modality == 'mv':
                        sampled_frames = mploader(self.video_prefix, vid_subpath, sampled_idxs, self.modality, self.accumulate, self.ds_factor, self.mv_minmaxnorm, self.mv_loadimg)
                    elif self.modality == 'res':
                        sampled_frames = mploader(self.video_prefix, vid_subpath, sampled_idxs, self.modality, self.accumulate, self.ds_factor, self.mv_minmaxnorm, self.mv_loadimg) 
                    elif self.modality == 'I':
                        sampled_frames = mploader(self.video_prefix, vid_subpath, sampled_idxs, self.modality, self.accumulate, self.ds_factor, self.mv_minmaxnorm, self.mv_loadimg)
                    elif self.modality == 'flow+mp4':
                        sampled_frames = mploader(self.video_prefix, vid_subpath, sampled_idxs, self.modality, self.accumulate, self.ds_factor, self.mv_minmaxnorm, self.mv_loadimg)
                    else:
                        sampled_frames = loader(video_path, (image_indices[sampled_idxs]).tolist(), self.modality)
                    if not sampled_frames is None:
                        successfule_trial = True
                        break

            else:   
                with Video(vid_path=video_path) as video:
                    if frame_count < 0:
                        frame_count = video.count_frames(check_validity=False)
                    for i_trial in range(20):
                        # dynamic sampling
                        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=v_id, prev_failed=(i_trial>0))
                        if set(list(sampled_idxs)).intersection(faulty_frames):
                            continue
                        prev_sampled_idxs = list(sampled_idxs)
                        # extracting frames
                        sampled_frames = video.extract_frames(idxs=sampled_idxs, force_color=self.force_color)
                        if sampled_frames is None:
                            faulty_frames.append(video.faulty_frame)
                        else:
                            successfule_trial = True
                            break
        except IOError as e:
            logging.warning(">> I/O error({0}): {1}".format(e.errno, e.strerror))
        if not successfule_trial:
            assert (self.backup_item is not None), \
                "VideoIter:: >> frame {} is error & backup is inavailable. [{}]'".format(faulty_frames, video_path)
            logging.warning(">> frame {} is error, use backup item! [{}]".format(faulty_frames, video_path))
            with Video(vid_path=self.backup_item['video_path']) as video:
                sampled_frames = video.extract_frames(idxs=self.backup_item['sampled_idxs'], force_color=self.force_color)
        elif self.tolerant_corrupted_video:
            # assume the error rate less than 10%
            if (self.backup_item is None) or (self.rng.rand() < 0.1):
                self.backup_item = {'video_path': video_path, 'sampled_idxs': sampled_idxs}

        clip_input = np.concatenate(sampled_frames, axis=2)
        if self.modality == 'rgb+flow':
            flow_input = np.concatenate(flow, axis=2)
        # apply video augmentation
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)
            if self.modality == 'rgb+flow':
                flow_input = self.transform_flow(flow_input)
        if self.modality == 'rgb+flow':
            return clip_input, flow_input, label, vid_subpath
        else:
            return clip_input, label, vid_subpath


    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                if self.modality == 'rgb+flow':
                    clip_input, flow_input, label, vid_subpath = self.getitem_from_raw_video(index)
                    
                else:
                    clip_input, label, vid_subpath = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                index = self.rng.choice(range(0, self.__len__()))
                logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        if self.return_item_subpath:
            if self.modality == 'rgb+flow':
                return clip_input, flow_input, label, vid_subpath
            else:
                return clip_input, label, vid_subpath
        else:
            return clip_input, label


    def __len__(self):
        return len(self.video_list)


    def _get_video_list(self,
                        video_prefix,
                        txt_list,
                        check_video=False,
                        cached_info_path=None):
        # formate:
        assert os.path.exists(video_prefix), "VideoIter:: failed to locate: `{}'".format(video_prefix)
        assert os.path.exists(txt_list), "VideoIter:: failed to locate: `{}'".format(txt_list)

        # try to load cached list
        cached_video_info = {}
        print(cached_info_path)
        if cached_info_path:
            if os.path.exists(cached_info_path):
                f = open(cached_info_path, 'r')
                cached_video_prefix = f.readline().split()[1]
                cached_txt_list = f.readline().split()[1]
                if (cached_video_prefix == video_prefix.replace(" ", "")) \
                    and (cached_txt_list == txt_list.replace(" ", "") or cached_txt_list == txt_list[1:].replace(" ", "")):
                    logging.info("VideoIter:: loading cached video info from: `{}'".format(cached_info_path))
                    lines = f.readlines()
                    for line in lines:
                        video_subpath, frame_count = line.split()
                        cached_video_info.update({video_subpath: int(frame_count)})
                else:
                    logging.warning(">> Cached video list mismatched: " +
                                    "(prefix:{}, list:{}) != expected (prefix:{}, list:{})".format(\
                                    cached_video_prefix, cached_txt_list, video_prefix, txt_list))
                f.close()
            else:
                if not os.path.exists(os.path.dirname(cached_info_path)):
                    os.makedirs(os.path.dirname(cached_info_path))
        # building dataset
        video_list = []
        new_video_info = {}
        logging_interval = 100
        with open(txt_list) as f:
            lines = f.read().splitlines()
            logging.info("VideoIter:: found {} videos in `{}'".format(len(lines), txt_list))
            for i, line in enumerate(lines):
                v_id, label, video_subpath = line.split()
                video_path = video_prefix
                if self.load_from_img:
                    path = video_subpath.split('/')
                    actual_path = os.listdir(os.path.join(video_prefix, path[0]))
                    
                    split_name = ''.join([j for j in video_subpath[len(path[0]):len(video_subpath) - 4] if j.isalnum()])
                    for i, name in enumerate(actual_path):
                        if ''.join([j for j in name if j.isalnum()]) == split_name:
                            video_subpath = os.path.join(path[0], name)
                            break
                if video_subpath.endswith('.avi'):
                    video_subpath = video_subpath[:len(video_subpath) - 4]
                video_path = os.path.join(video_prefix, video_subpath)
                
                if not os.path.exists(video_path):
                    logging.warning("VideoIter:: >> cannot locate `{}'".format(video_path))
                    continue
                if check_video:
                    if video_subpath in cached_video_info:
                        frame_count = cached_video_info[video_subpath]
                    elif video_subpath in new_video_info:
                        #print(new_video_info,video_subpath)
                        frame_count = new_video_info[video_subpath]
                    else:
                        frame_count = -1
                        if self.load_from_img:
                            image_indices = []
                            if self.modality == 'rgb':
                                for image_file_name in os.listdir(video_path):
                                    if 'img' not in image_file_name:
                                        continue
                                    image_indices.append(int(image_file_name[4:9]))
                            elif self.modality == 'flow' or self.modality == 'rgb+flow':
                                for image_file_name in os.listdir(video_path):
                                    if 'flow_x' not in image_file_name:
                                        continue
                                    image_indices.append(int(image_file_name[7:12]))
                            

                            if len(image_indices) == 0:
                                print('no image files', video_path)
                                frame_count = 0
                            else:
                                image_indices.sort(reverse=True)
                                frame_count = image_indices[0]
                        else:
                            frame_count = self.video.open(video_path).count_frames(check_validity=True)
                        new_video_info.update({video_subpath: frame_count})
                else:
                    frame_count = -1
                info = [int(v_id), int(label), video_subpath, frame_count]
                video_list.append(info)
                if check_video and (i % logging_interval) == 0:
                    logging.info("VideoIter:: - Checking: {:d}/{:d}, \tinfo: {}".format(i, len(lines), info))

        # caching video list
        if cached_info_path and len(new_video_info) > 0:
            logging.info("VideoIter:: adding {} lines new video info to: {}".format(len(new_video_info), cached_info_path))
            cached_video_info.update(new_video_info)
            with open(cached_info_path, 'w') as f:
                # head
                f.write("video_prefix: {:s}\n".format(video_prefix.replace(" ", "")))
                f.write("txt_list: {:s}\n".format(txt_list.replace(" ", "")))
                # content
                for i, (video_subpath, frame_count) in enumerate(cached_video_info.items()):
                    if i > 0:
                        f.write("\n")
                    f.write("{:s}\t{:d}".format(video_subpath, frame_count))

        return video_list
    
