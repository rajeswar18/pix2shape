import datetime
import numpy as np
import os


# FROM parameters_halfbox_shapenet
class opt:
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    batch_size = 2 # Number of views to generate
    render_reset_freq = 100 # Number of renders after which reset the scene to avoid accumulating errors
    # Image
    width=128
    height = 128
    # Camera
    def_cam_dist = 0.8
    cam_dist = 8.
    new_dist_r = cam_dist*.6
    new_dist_g = cam_dist*.7
    angle = 30 # camera angle (don't need b/c lookat!)
    fovy = 26 # Field of view in the vertical direction.
    focal_length = 0.1
    theta_range = [20, 80]
    phi_range = [20, 70]
    axis = [0., 1., 0.]
    lookat = [0., 0., 0.]
    # Object
    plane_size = 4
    new_max_dim = cam_dist/5
    # obj_location = (.7, .7, 0.2)
    obj_location = (.3, .3, 0.2)
    # Lights
    n_lights = 3
    light_pos = [None, (0.2*new_dist_r, 0.6*new_dist_r, 0.8*new_dist_r), (0.8*new_dist_g, 0.8*new_dist_g, 0.6*new_dist_g)]
    rn_light_pos_dist = cam_dist*.7
    light_color = [(0.8, 0.8, 0.8), (0.8, 0.1, 0.1), (0.2, 0.8, 0.2)]
    light_attn_dist = cam_dist*1.6
    # Render
    splats_img_size = 128
    pixel_samples = 1
    # Training
    name = "exp1"
    lr = 2e-4
    disc_type = "cnn"
    gz_gi_loss = 0.2
    est_normals = True
    zloss = 0.05
    unit_normalloss = 0.0
    normal_consistency_loss_weight = 10.0
    spatial_var_loss_weight = 0.0
    grad_img_depth_loss = 0.0
    spatial_loss_weight = 0.0
    same_view = False
    full_sphere_sampling = False
    # obj_filepath = "/home/user1/Downloads/einstein.obj"
    # obj_filepath = "/home/user1/Downloads/einstein_10000.obj"
    obj_filepath = "/home/user1/GitHubRepos/pix2scene/data/cube/cube.obj"
    # To save rendered images
    save_image = True
    save_dir = "/home/user1/blender_experiments/{0:%Y%m%d_%H%M%S}_einstein_10000".format(datetime.datetime.now())
