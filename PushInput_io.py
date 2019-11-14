import pickle
import tensorflow as tf
from munch import Munch
import os.path as path
import os

from TFDataInput import TFDataInput

import ipdb
st = ipdb.set_trace

import numpy as np
def preprocess_color(x):
    return tf.cast(x,tf.float32) * 1./255 - 0.5


dataroot = "/projects/katefgroup/datasets/bullet_push_frontonly"
dataset_location = f"{dataroot}/1112_1obj"
data_paths = {
              "train": f"{dataroot}/1112_1objt.txt", 
              "val": f"{dataroot}/1112_1objv.txt", 
              "test": f"{dataroot}/1112_1objv.txt"
            }
B = 2
DEBUG_MODE = False


class PushInput_io(TFDataInput):
    def __init__(self):
        super().__init__()

        # all data share the same basic info
        self.load_basic_info(dataset_location)
        
        self.train_data = self.make_data(data_paths["train"], True)
        self.val_data = self.make_data(data_paths["val"], True)
        self.test_data = self.make_data(data_paths["test"], False, True)

    def load_basic_info(self, folder):
        with open(path.join(folder, "cam_info.pkl"), "rb") as f:
            camera_info = pickle.load(f)
            self.basic_info = pickle.load(f)

    def make_data(self, fn, shuffle, istest = False):
        with open(fn, 'r') as f:
            content = f.readlines()
          
        records = [dataset_location + '/' + line.strip() for line in content]
        nRecords = len(records)
        for record in records:
            assert(os.path.isfile(record), f'Record at {record} was not found')
        print(f'found {nRecords} records in {dataset_location}')

        data = tf.data.TFRecordDataset(
            records,
            compression_type="GZIP"
        )
        nr = 4
        if not shuffle:
            nr = 1 

        data = data.map(self.decode , num_parallel_calls = nr)
        data = data.map(self.pack, num_parallel_calls = nr)
        if not istest and shuffle and not DEBUG_MODE:
            # tfrecords already shuffled, put a small buffer size here just in case
            data = data.shuffle(buffer_size=20)
            data = data.repeat()

        data = data.batch(B)

        if not istest:
            data = data.prefetch(16)
        iterator = tf.compat.v1.data.make_one_shot_iterator(data)
        return iterator
         
    def decode(self, example):
        T_TRAJ = self.basic_info["T"]
        N_CAM = self.basic_info["num_views"]
        H_data = self.basic_info["image_height"]
        W_data = self.basic_info["image_width"]

        Nobj = self.basic_info["N_MAX_OBJECTS"]
        Nagent = self.basic_info["N_MAX_AGENT_PARTS"]
        action_dim = self.basic_info["action_dim"]
        voxel_size = self.basic_info["voxel_size"]

        keys_to_features={
            'rgb_camXs_raw': tf.io.FixedLenFeature([], tf.string),
            'depth_camXs_raw': tf.io.FixedLenFeature([], tf.string),
            'seg_camXs_raw': tf.io.FixedLenFeature([], tf.string),
            'pix_T_cams_raw': tf.io.FixedLenFeature([], tf.string),
            'origin_T_camXs_raw': tf.io.FixedLenFeature([], tf.string),
            'xyzorn_objects': tf.io.FixedLenFeature([], tf.string),
            'object_class': tf.io.FixedLenFeature([], tf.string),
            'xyzorn_agent': tf.io.FixedLenFeature([], tf.string),
            'actions': tf.io.FixedLenFeature([], tf.string),
            'voxels_objects': tf.io.FixedLenFeature([], tf.string),
            'resize_factor_objects': tf.io.FixedLenFeature([], tf.string),
            'voxels_agent_parts': tf.io.FixedLenFeature([], tf.string),
            'resize_factor_agent_parts': tf.io.FixedLenFeature([], tf.string),
            'raw_seq_filename': tf.io.FixedLenFeature([], tf.string)
        }

        parsed = tf.io.parse_single_example(
          serialized=example,
          # Defaults are not specified since both keys are required.
          features=keys_to_features
        )

        rgb_camXs = tf.io.decode_raw(parsed['rgb_camXs_raw'], tf.uint8)
        rgb_camXs = tf.reshape(rgb_camXs, [T_TRAJ, N_CAM, H_data, W_data, 4])[..., :3]
        rgb_camXs = preprocess_color(rgb_camXs)

        depth_camXs = tf.io.decode_raw(parsed['depth_camXs_raw'], tf.float32)
        depth_camXs = tf.reshape(depth_camXs, [T_TRAJ, N_CAM, H_data, W_data, 1])

        seg_camXs = tf.io.decode_raw(parsed['seg_camXs_raw'], tf.int32)
        seg_camXs = tf.reshape(seg_camXs, (T_TRAJ, N_CAM, H_data, W_data, 1))

        pix_T_cams = tf.io.decode_raw(parsed['pix_T_cams_raw'], tf.float32)
        pix_T_cams = tf.reshape(pix_T_cams, [1, N_CAM, 4, 4])
        pix_T_cams = tf.tile(pix_T_cams, [T_TRAJ, 1, 1, 1])
        
        origin_T_camXs = tf.io.decode_raw(parsed['origin_T_camXs_raw'], tf.float32)
        origin_T_camXs = tf.reshape(origin_T_camXs, [T_TRAJ, N_CAM, 4, 4])
        # origin_T_camXs = tf.tile(origin_T_camXs, [T_TRAJ, 1, 1, 1])

        xyzorn_objects = tf.io.decode_raw(parsed['xyzorn_objects'], tf.float32)
        xyzorn_objects = tf.reshape(xyzorn_objects, (T_TRAJ, Nobj, 7 + 6))

        object_class = tf.io.decode_raw(parsed['object_class'], tf.float32)
        object_class = tf.reshape(object_class, (T_TRAJ, Nobj))

        xyzorn_agent = tf.io.decode_raw(parsed['xyzorn_agent'], tf.float32)
        xyzorn_agent = tf.reshape(xyzorn_agent, (T_TRAJ, Nagent, 7 + 6))

        actions = tf.io.decode_raw(parsed['actions'], tf.float32)
        actions = tf.reshape(actions, (T_TRAJ-1, action_dim))

        voxels_objects = tf.io.decode_raw(parsed['voxels_objects'], tf.uint8)
        voxels_objects = tf.reshape(voxels_objects, (Nobj, voxel_size, voxel_size, voxel_size))
        voxels_agent_parts = tf.io.decode_raw(parsed['voxels_agent_parts'], tf.uint8)
        voxels_agent_parts = tf.reshape(voxels_agent_parts, (Nagent, voxel_size, voxel_size, voxel_size))

        resize_factor_objects = tf.io.decode_raw(parsed['resize_factor_objects'], tf.float32)
        resize_factor_objects = tf.reshape(resize_factor_objects, (Nobj, 3))
        resize_factor_agent_parts = tf.io.decode_raw(parsed['resize_factor_agent_parts'], tf.float32)
        resize_factor_agent_parts = tf.reshape(resize_factor_agent_parts, (Nagent, 3))

        raw_seq_filename = parsed["raw_seq_filename"]

        return rgb_camXs, depth_camXs, seg_camXs, pix_T_cams, origin_T_camXs, xyzorn_objects, object_class, xyzorn_agent, actions, voxels_objects, voxels_agent_parts, resize_factor_objects, resize_factor_agent_parts, raw_seq_filename

    def clip_data(self, item, start_t, len):
        item = tf.reshape(item[start_t:start_t + len, ...],
                          [len] + item.shape[1:].as_list())
        return item

    def pack(self, rgb_camXs, depth_camXs, seg_camXs, pix_T_cams, origin_T_camXs,
            xyzorn_objects, object_class, xyzorn_agent, actions, 
            voxels_objects, voxels_agent_parts, resize_factor_objects, 
            resize_factor_agent_parts, raw_seq_filename):
        names = ['rgb_camXs', 'depth_camXs', 'seg_camXs', 'pix_T_cams', 'origin_T_camXs', 'xyzorn_objects', 'object_class', 'xyzorn_agent', 'actions', 'voxels_objects', 'voxels_agent_parts', 'resize_factor_objects', 'resize_factor_agent_parts', 'raw_seq_filename']
        stuff = [rgb_camXs, depth_camXs, seg_camXs, pix_T_cams, origin_T_camXs, xyzorn_objects, object_class, xyzorn_agent, actions, voxels_objects, voxels_agent_parts, resize_factor_objects, resize_factor_agent_parts, raw_seq_filename]
        return Munch(zip(names, stuff))