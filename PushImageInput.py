import tensorflow as tf
import numpy as np
import pyquaternion as pyq
import tfquaternion as tfq
import scipy

import utils
from munch import Munch
from PushInput_io import PushInput_io



import ipdb
st=ipdb.set_trace

class Config:
    def __init__(self):
        self.NUM_VIEWS=2 # number of views used as inputs
        self.NUM_PREDS=1 # number of predicted views

        self.N_VIEW = self.NUM_VIEWS + self.NUM_PREDS
        self.N_CAM = 27
        self.H = 128
        self.W = 128
        self.T_TRAJ = 6 # length of trajectory in input data
        self.V = 20000
        self.MIN_DEPTH_RANGE = 5
        self.MAX_DEPTH_RANGE = 21

class PushImageInput:
    def __init__(self, datamod):
        self.data_io = PushInput_io(datamod)
        self.config = Config()
    
    def data(self, index):
        inputs = self.data_io.data(index)
        inputs = self.get_front_and_side(inputs)

        inputs = self.extract_rand_views(inputs)
        inputs.rgb_camXs = self.resize(inputs.rgb_camXs)
        inputs = self.misc(inputs)

        data = Munch(inputs=self.make_inputs(inputs))
        return data

    def get_delta(self, xyzorn, xyzorn_after):

        xyz, orn, vel = self.split_states(xyzorn, mode="h13")
        xyz_after, orn_after, vel_after = self.split_states(xyzorn_after, mode="h13")

        delta_xyz = xyz_after - xyz
        quat1 = tfq.Quaternion(orn)
        quat2 = tfq.Quaternion(orn_after)

        delta_quat = quat2*quat1.inverse()

        delta_vel =  vel_after - vel
        return tf.concat([delta_xyz, delta_quat, delta_vel], 3)

    def split_states(self, states, mode="h13"):
        """
        if mode == "obj_agent":
            return states[..., :self.nobjs, :], states[..., self.nobjs:, :]
        """
        if mode == "h13":
            # xyz(3), orn(4), velv(6)
            return states[..., :3], states[..., 3:7], states[..., 7:]
        else:
            raise Exception(f"data format is not supported: {mode}")

    def make_inputs(self, data):
        # use only the first N_VIEWS as inputs
        N_INPUT_VIEWS = self.config.N_VIEW - 1
        T_TRAJ_1 = self.config.T_TRAJ - 1

        return Munch(rgb_camXs=data["rgb_camXs"][:, :T_TRAJ_1, :N_INPUT_VIEWS],
                     rgbvalid_camXs=data["rgbvalid_camXs"][:, :T_TRAJ_1, :N_INPUT_VIEWS],
                     pix_T_cams = data["pix_T_cams"][:, :T_TRAJ_1, :N_INPUT_VIEWS],
                     origin_T_camXs = data["origin_T_camXs"][:, :T_TRAJ_1, :N_INPUT_VIEWS],
                     xyzorn_objects = data["xyzorn_objects"][:, :T_TRAJ_1, :, :], #B X T_TRAJ_1 X N_OBJS X 13
                     object_class = data["object_class"][:, :T_TRAJ_1, :], #B X T_TRAJ_1 X N_OBJS
                     xyzorn_agents = data["xyzorn_agent"][:, :T_TRAJ_1, :, :], #B X T_TRAJ_1 X N_AGENTS X 13
                     actions = data["actions"], #B X T_TRAJ_1
                     voxels = tf.expand_dims(tf.transpose(tf.cast(data["voxels_objects"], dtype=tf.float32), [0, 1, 3, 2, 4]), axis=-1),
                     voxels_agent_parts = tf.expand_dims(tf.transpose(tf.cast(data["voxels_agent_parts"], dtype=tf.float32), [0, 1, 3, 2, 4]), axis=-1),
                     # B X N_OBJS X H X W X D x 1
                     resize_factor = data["resize_factor_objects"], #B X N_OBJS X 3
                     resize_factor_agent_parts = data["resize_factor_agent_parts"], #B X N_OBJS X 3
                    )


    def make_target(self, data):
        N_INPUT_VIEWS = self.config.NUM_VIEWS
        T_TRAJ_1 = self.config.T_TRAJ - 1


        delta_objs_state = self.get_delta(data["xyzorn_objects"][:, :-1, :, :], data["xyzorn_objects"][:, 1:, :, :])
        delta_agent_state = self.get_delta(data["xyzorn_agent"][:, :-1, :, :], data["xyzorn_agent"][:, 1:, :, :])

        return Munch(rgb_camXs=data["rgb_camXs"][:, :, N_INPUT_VIEWS:],
                rgbvalid_camXs=data["rgbvalid_camXs"][:, :, N_INPUT_VIEWS:],
                pix_T_cams = data["pix_T_cams"][:, :, N_INPUT_VIEWS:],
                origin_T_camXs = data["origin_T_camXs"][:, :, N_INPUT_VIEWS:],
                xyzorn_objects = data["xyzorn_objects"][:, 1:, :, :], #B X T_TRAJ_1 X N_OBJS X 13
                object_class = data["object_class"][:, 1:, :], #B X T_TRAJ_1 X N_OBJS
                xyzorn_agents = data["xyzorn_agent"][:, 1:, :, :], #B X T_TRAJ_1 X N_AGENTS X 13
                delta_xyzorn_objects = delta_objs_state,
                delta_xyzorn_agents = delta_agent_state
               )
    def make_extra(self, data):
        N_INPUT_VIEWS = self.config.NUM_VIEWS
        return Munch(rgb_camFront=data["rgb_CamFront"],
                     rgb_camSide=data["rgb_CamSide"])



    def get_front_and_side(self, inputs):
        B, T_TRAJ, N_CAM, H, W, C = inputs["rgb_camXs"].shape
        images = inputs["rgb_camXs"]
        images_front = images[:, :, 13, :, :, :3]
        images_side = images[:, :, 8, :, :, :3]

        B, T, H_data, W_data, C = images_front.shape
        H = self.config.H
        W = self.config.W

        def resize(rgb_camXs):
            if H != H_data or W != W_data:
                rgb_camXs = tf.reshape(rgb_camXs, [B*T, H_data, W_data, 3])
                rgb_camXs = tf.image.resize(rgb_camXs, [H, W])
                rgb_camXs = tf.reshape(rgb_camXs, [B, T, H, W, 3])
            return rgb_camXs
        inputs.rgb_CamFront = resize(images_front)
        inputs.rgb_CamSide = resize(images_side)
        return inputs


    def visualize_data(self, data_, batch_id=0):

        import matplotlib
        matplotlib.use('tkagg') # uncomment for interactive visualization
        import matplotlib.pyplot as plt

        import numpy as np
        batch_id = 0
        B, T_TRAJ, N_VIEWS, _, _= data_["origin_T_camXs"].shape

        #fig, ax = utils.pyplot_vis.plot_voxel(data_["voxels"][0, 0, :, :, :], fig_id=4, coord="xright-yback")

        from utils.memcoord import Coord, VoxCoord
        from utils.protos import VoxProto

        seg_coord = Coord(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        seg_proto = VoxProto([128, 128, 128])
        seg_voxcoord = VoxCoord(seg_coord, seg_proto)



        world_coord = Coord(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
        world_proto = VoxProto([80, 80, 80])
        world_voxcoord = VoxCoord(world_coord, world_proto)


        object_scale = data_["resize_factor"]
        object_scale_mat = tf.linalg.diag(object_scale)
        objectworldsize_T_object1x1 = utils.geom.mat_3x3_to_4x4(object_scale_mat)

        #t = 0
        for t in range(0, 1): #6, 2):
            world_T_obj = utils.basic.object_xyzorn_to_4x4matrix(data_["xyzorn_objects"][:, t])
            world_T_obj_ = tf.matmul(world_T_obj, objectworldsize_T_object1x1)
            obj_T_world_ = tf.linalg.inv(world_T_obj_)

            voxels_f32 = tf.expand_dims(tf.transpose(tf.cast(data_["voxels"], dtype=tf.float32), [0, 1, 3, 2, 4]), axis=-1)
            #fig, ax = utils.pyplot_vis.plot_adam_voxel(voxels_f32[0, 0, :, :, :, 0], fig_id=4, coord="xright-ydown")

            scene_mask, _ = utils.voxelizer.resample_to_target_views(voxels_f32, seg_voxcoord, obj_T_world_, world_voxcoord)
            fig2, ax2 = utils.pyplot_vis.plot_adam_voxel(tf.reduce_sum(scene_mask[0, :, :, :, :,0], axis=0), fig_id=5 +t , title=f"t={t}", coord="xright-ydown")

        # crop zoom
        zoom_coord = Coord(-0.405, 0.405, -0.405, 0.405, -0.405, 0.405)
        zoom_proto = VoxProto([64, 64, 64])
        zoom_voxcoord = VoxCoord(zoom_coord, zoom_proto)

        bullet_T_adam = np.zeros((4,4), dtype=np.float32)
        bullet_T_adam[0,0] = 1
        bullet_T_adam[1,2] = 1
        bullet_T_adam[2,1] = -1
        bullet_T_adam[3,3] = 1


        B, NOBJS, _, _ = world_T_obj.shape
        bullet_T_adam = tf.tile(tf.reshape(tf.convert_to_tensor(bullet_T_adam, dtype=tf.float32), [1,1,4,4]), [B, NOBJS, 1, 1])
        world_T_objadam = tf.matmul(world_T_obj, bullet_T_adam)
        world_T_objadam_ = utils.geom.reset_ornmat(world_T_objadam)

        scene_mask = tf.tile(tf.reduce_sum(scene_mask, axis=1, keepdims=1), [1, 2, 1,1,1,1])
        crop_mask, _ = utils.voxelizer.resample_to_target_views(scene_mask, world_voxcoord, world_T_objadam_, zoom_voxcoord)

        _, _ = utils.pyplot_vis.plot_adam_voxel(crop_mask[0, 0, :, :, :,0], fig_id=10, coord="xright-ydown")
        _, _ = utils.pyplot_vis.plot_adam_voxel(crop_mask[0, 1, :, :, :,0], fig_id=11, coord="xright-ydown")

        plt.show()
        # st()



        V = self.config.V

        T_TRAJ_SHOW= min(T_TRAJ, 2)
        fig_pcd = None
        fig_obj = None

        #st()
        for t in range(0, T_TRAJ_SHOW, 1):
            print("t", t)
            # print only object 0s
            print("object_pos", data_.xyzorn_objects[batch_id, t, 0:1])
            xyzorn_object = data_.xyzorn_objects[batch_id, t, 0:1]
            object_orn_mat = utils.basic.object_xyzorn_to_4x4matrix(xyzorn_object)
            #print("orn_mat:")

            #if t == 0:
            fig_pcd, ax_pcd = utils.pyplot_vis.plot_cam(data_["origin_T_camXs"][batch_id, t], length=0.5, ylims=[-5, 0], xlims = [-2.5, 2.5], zlims = [-2.5, 2.5], msg="train_",\
                                                      fig_id=1, title=f"t={t}", fig=fig_pcd, subplot=100+T_TRAJ_SHOW*10+(t+1), coord="xright-ydown")

            fig_obj, ax_obj = utils.pyplot_vis.plot_cam(data_["origin_T_camXs"][batch_id, t], length=0.5, ylims=[-3, 0], xlims = [-1.5, 1.5], zlims = [-1.5, 1.5], msg="train_",\
                                                      fig_id=2, title=f"t={t}", fig=fig_obj, subplot=100+T_TRAJ_SHOW*10+(t+1), coord="xleft-ydown")

            _, ax_obj = utils.pyplot_vis.plot_cam(object_orn_mat, length=1.0, msg="train_",\
                                                       ax=ax_obj)

            # compute xyz_camRs
            origin_T_camXs_n = data_["origin_T_camXs"][batch_id, t]
            for view_id in range(1):#N_VIEWS):
                print("cam_T_center dist", np.linalg.norm(origin_T_camXs_n[view_id, :3, 3] - np.array([0, -0.5, 0])))
                xyz_camXs_n = data_["xyz_camXs"][batch_id, t]
                xyz_camRs_n = xyz_camRs = utils.geom.apply_4x4(origin_T_camXs_n, xyz_camXs_n)
                subsampled_xyz_camRs_n = np.stack([xyz_camRs_n.numpy()[view_id, list(range(0, V, 20)), :] for view_id in range(N_VIEWS)], axis=0)
                _, ax_pcd  = utils.pyplot_vis.plot_pointclouds(subsampled_xyz_camRs_n, ax=ax_pcd)

                #fig[t + T_TRAJ], ax[t + T_TRAJ] = utils.pyplot_vis.plot_cam(data_["origin_T_camXs"][batch_id, t], ylims=[-5, 0], xlims = [-2.5, 2.5], zlims = [-2.5, 2.5], msg="train_", coord = "xleft-ydown", fig_id=t + T_TRAJ, length=0.5)
                #_, ax[t + T_TRAJ]  = utils.pyplot_vis.plot_pointclouds(subsampled_xyz_camRs_n, ax=ax[t + T_TRAJ])

            #imgplot = utils.pyplot_vis.plot_images((data_["rgb_camXs"][batch_id,t,:,:,:, :] + 0.5), fig_id=2 * T_TRAJ + t)
        fig_img = None
        T_TRAJ_SHOW= min(T_TRAJ, 6)
        for t in range(T_TRAJ_SHOW):
            fig_img, _ = utils.pyplot_vis.plot_images((data_["rgb_camXs"][batch_id,t,:,:,:, :] + 0.5), fig_id=3, title=f"t={t}", fig=fig_img, subplot=100+T_TRAJ_SHOW*10+(t+1))
        #plt.show()
        #st()
        print("hello")

    def bulletxyz_to_adamxyz(self, xyzorn_btn):
        import numpy as np
        B, T_TRAJ, N_OBJS, _ = xyzorn_btn.shape
        xyz = xyzorn_btn[:,:,:,:3]
        orn = xyzorn_btn[:,:,:,3:7]
        xyzvl = xyzorn_btn[:,:,:,7:10]
        xyzva = xyzorn_btn[:,:,:,10:13]

        adam_T_bullet = tf.convert_to_tensor(np.array([[1,0,0], [0,0,-1], [0,1,0]], dtype=np.float32), dtype=tf.float32)
        adam_T_bullet = tf.tile(tf.reshape(adam_T_bullet, [1, 1, 1, 3, 3]), [B, T_TRAJ, N_OBJS, 1, 1])

        adam_T_bullet_ = np.array([[1,0,0], [0,0,-1], [0,1,0]], dtype=np.float64)

        adam_T_bullet_64 = tf.convert_to_tensor(np.array([[1,0,0], [0,0,-1], [0,1,0]], dtype=np.float64), dtype=tf.float64)
        adam_T_bullet_64 = tf.tile(tf.reshape(adam_T_bullet_64, [1, 1, 1, 3, 3]), [B, T_TRAJ, N_OBJS, 1, 1])

        xyz_adam = tf.squeeze(tf.matmul(adam_T_bullet, tf.expand_dims(xyz, -1)), axis=-1)
        xyzvl_adam = tf.squeeze(tf.matmul(adam_T_bullet, tf.expand_dims(xyzvl, -1)), axis=-1)
        xyzva_adam = tf.squeeze(tf.matmul(adam_T_bullet, tf.expand_dims(xyzva, -1)), axis=-1)


        orn_mat_bullet = tfq.Quaternion(orn).as_rotation_matrix()
        orn_mat_adam = tf.matmul(tf.matmul(adam_T_bullet, orn_mat_bullet), tf.cast(tf.linalg.inv(adam_T_bullet_64), dtype=tf.float32))
        orn_mat_adam = orn_mat_adam.numpy()
        orn_adam_b_t_objs = []
        for batch_id in range(B):
            orn_adam_t_objs = []
            for t in range(T_TRAJ):
                orn_adam_objs = []
                for obj_id in range(N_OBJS):
                    orn = xyzorn_btn[batch_id, t, obj_id,3:7].numpy()
                    orn_mat_bullet = pyq.Quaternion(orn).rotation_matrix
                    orn_mat_adam = np.matmul(np.matmul(adam_T_bullet_, orn_mat_bullet), np.linalg.inv(adam_T_bullet_))

                    orn_adam = pyq.Quaternion(matrix=orn_mat_adam)
                    orn_adam_objs.append(np.array(list(orn_adam), dtype=np.float32))
                orn_adam_objs = np.stack(orn_adam_objs, axis=0)
                orn_adam_t_objs.append(orn_adam_objs)
            orn_adam_t_objs = np.stack(orn_adam_t_objs, axis=0)
            orn_adam_b_t_objs.append(orn_adam_t_objs)
        orn_adam_b_t_objs = np.stack(orn_adam_b_t_objs, axis=0)
        orn_adam_b_t_objs = tf.convert_to_tensor(orn_adam_b_t_objs, dtype=tf.float32)
        xyzorn_adam = tf.concat([xyz_adam, orn_adam_b_t_objs, xyzvl_adam, xyzva_adam], axis=3)
        return xyzorn_adam



    def xyzorn_transform_inv(self, xyzorn_btn, xyz_class=None, old_origin=[0,0,0.5]):
        import numpy as np
        """quat90_zyx = pyq.Quaternion(matrix=rot_zyx)
        xyzorn: B x 13
        This function transform the xyz in the bullet form (z pointing up, x pointing left, and y pointing front)
        to 3d tensor format (order is zyx, y pointing up, x pointing to the left, z pointing the back)
        centensform(xyzorn, xyz_class=None, origin = [0,0,0]):
        """
        # rotate 90 degree along z-y plan, change x,y order

        B, T_TRAJ, N_OBJS, _ = xyzorn_btn.shape

        __pB = lambda x: utils.basic.pack_seqdim(x, B)
        __uB = lambda x: utils.basic.unpack_seqdim(x, B)

        __pT = lambda x: utils.basic.pack_seqdim(x, B * T_TRAJ)
        __uT = lambda x: utils.basic.unpack_seqdim(x, B * T_TRAJ)

        xyzorn_btn_ = __pB(xyzorn_btn)
        xyzorn_btn__ = __pT(xyzorn_btn_)
        if xyz_class is not None:
            xyz_class_ = __pB(xyz_class)
            xyz_class__ = __pT(xyz_class_)

        batch_size, dim = xyzorn_btn__.shape

        #import ipdb; ipdb.set_trace()
        rotate_yx_90 = np.array([[1,0,0], [0,0,1], [0,-1,0]])
        #xyz = xyzorn[:,:, :3] - np.array(origin, dtype=np.float32)    

        outputs = []
        for batch_id in range(B * T_TRAJ * N_OBJS):
            orn = xyzorn_btn__[batch_id,3:7]
            xyz = xyzorn_btn__[batch_id,:3]
            xyzvl = xyzorn_btn__[batch_id,7:10]
            xyzva = xyzorn_btn__[batch_id,10:13]

            quat90_zyx = pyq.Quaternion(orn).rotation_matrix
            rot90 = np.reshape(np.flip(np.reshape(quat90_zyx, [-1])), [3, 3])
            rot = np.matmul(np.linalg.inv(rotate_yx_90), rot90)
            new_quat = list(pyq.Quaternion(matrix=rot))        
            new_orn = new_quat 

            #this is for bullet orn representation: np.concatenate([new_quat[1:], new_quat[:1]], 0)

            xyz = np.array([xyz[2], -xyz[0], xyz[1]])
            new_xyz = xyz + np.array(old_origin)
            new_xyzvl = np.array([xyzvl[2], -xyzvl[0], xyzvl[1]]) 
            new_xyzva = np.array([xyzva[2], -xyzva[0], xyzva[1]]) 
            new_out = np.concatenate([new_xyz, new_orn, new_xyzvl, new_xyzva], 0)

            if xyz_class is not None and xyz_class__[batch_id] == 0:
                new_out = np.array([0,0,0, 1, 0,0,0, 0, 0,0, 0,0,0])
            
            outputs.append(new_out)


        outputs = tf.stack(outputs, axis=0)
        xyzorn_btn__ = tf.cast(tf.convert_to_tensor(outputs, dtype=tf.float64), tf.float32)

        xyzorn_btn_ = __uT(xyzorn_btn__)
        xyzorn_btn = __uB(xyzorn_btn_)

        return xyzorn_btn

    def compute_xyz(self, depth_camXs_bn, pix_T_cams_bn):
        '''
        depth_camXs_batch: [B, N_VIEW, H, W, 1]
        pix_T_cams_batch: [B, N_VIEW, 4, 4]
        '''

        B, N_VIEW, H, W, _  = depth_camXs_bn.shape

        __p = lambda x: pack_seqdim(x, B)
        __u = lambda x: unpack_seqdim(x, B)

        V = self.config.V
        MIN_DEPTH_RANGE = self.config.MIN_DEPTH_RANGE
        MAX_DEPTH_RANGE = self.config.MAX_DEPTH_RANGE
        
        def clip(xyz_camXs_single):
            xyz_camXs_single = tf.gather_nd(xyz_camXs_single, tf.where(xyz_camXs_single[:, 2] > MIN_DEPTH_RANGE))
            xyz_camXs_single = tf.gather_nd(xyz_camXs_single, tf.where(xyz_camXs_single[:, 2] < MAX_DEPTH_RANGE))

            V_current = xyz_camXs_single.shape[0]
            if V_current > V:
                xyz_camXs_single = tf.random.shuffle(xyz_camXs_single)[:V]
            elif V_current < V:
                xyz_camXs_single = tf.pad(xyz_camXs_single, [[0, V-V_current], [0, 0]], 'CONSTANT')

            return xyz_camXs_single
            
        def depth2xyz(depth_camXs, pix_T_cams):

            """
            depth_camXs: B X H X W X 1

            """
            xyz_camXs = utils.geom.depth2pointcloud(depth_camXs, pix_T_cams)
            xyz_camXs = tf.map_fn(clip, xyz_camXs)
            return xyz_camXs

        depth_camXs_bn_ = __p(depth_camXs_bn)
        pix_T_cams_bn_ = __p(pix_T_cams_bn)

        xyz_camXs_bn_ = depth2xyz(depth_camXs_bn_, pix_T_cams_bn_)
        xyz_camXs_bn = __u(xyz_camXs_bn_)

        return xyz_camXs_bn

    def resize(self, rgb_camXs):
        B, T, N_VIEW, H_data, W_data, C = rgb_camXs.shape
        H = self.config.H
        W = self.config.W

        if H != H_data or W != W_data:
            rgb_camXs = tf.reshape(rgb_camXs, [B*T*N_VIEW, H_data, W_data, 3])
            rgb_camXs = tf.image.resize(rgb_camXs, [H, W])
            rgb_camXs = tf.reshape(rgb_camXs, [B, T, N_VIEW, H, W, 3])

        return rgb_camXs


    def extract_rand_views(self, inputs):
        B, T_TRAJ, N_CAM, H, W, C = inputs["rgb_camXs"].shape
        N_VIEW = self.config.N_VIEW
        idx = tf.random.shuffle(tf.range(N_CAM))[:N_VIEW]
        for k in inputs: #iterate through the batch
            if k in ['rgb_camXs', 'depth_camXs', 'pix_T_cams', 'origin_T_camXs']:
                inputs[k] = tf.gather(inputs[k], idx, axis=2)
        return inputs


    def misc(self, inputs):
        mask = tf.ones_like(inputs.rgb_camXs)
        inputs.rgbvalid_camXs = mask[..., :1]
        return inputs

def pack_seqdim(tensor, B):
    shapelist = tensor.shape.as_list()
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = tf.reshape(tensor, [B*S] + otherdims)
    return tensor
    
def unpack_seqdim(tensor, B):
    shapelist = tensor.get_shape().as_list()
    BS = shapelist[0]
    otherdims = shapelist[1:]
    assert(BS % B == 0)
    S = int(BS/B)
    tensor = tf.reshape(tensor, [B, S] + otherdims)
    return tensor
