import datetime
import os


# FROM parameters_halfbox_shapenet
class opt:
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    batch_size = 2 # Number of views to generate
    render_reset_freq = 100 # Number of renders after which reset the scene to avoid accumulating errors
    # Image
    width=128
    height = 128
    # Object
    plane_size = 4
    new_max_dim = 2.5
    # Camera
    def_cam_dist = 3.
    cam_dist = 8.
    r2b_ratio = cam_dist/def_cam_dist
    angle = 30 # camera angle (don't need b/c lookat!)
    fovy = 26 # Field of view in the vertical direction.
    focal_length = 0.1
    theta_range = [20, 80]
    phi_range = [20, 70]
    axis = [0., 1., 0.]
    lookat = [0., 0., 0.]
    # Lights
    n_lights = 3
    light_pos = [None, (0.2*r2b_ratio, 0.6*r2b_ratio, 0.8*r2b_ratio), (0.8*r2b_ratio, 0.8*r2b_ratio, 0.6*r2b_ratio)]
    rn_light_pos_dist = cam_dist
    light_color = [(0.8, 0.8, 0.8), (0.8, 0.1, 0.1), (0.2, 0.8, 0.2)]
    light_attn_dist = 14.142
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
    root_dir = "/path/pix2scene/data/cube"
    # obj_filepath = "/home/user1/Downloads/einstein.obj"
    obj_filepath = "/home/user1/Downloads/einstein_10000.obj"
    # obj_filepath = "/home/user1/GitHubRepos/pix2scene/data/cube/cube.obj"
    # To save rendered images
    save_image = True
    save_dir = "/home/user1/blender_experiments/{0:%Y%m%d_%H%M%S}_einstein_10000".format(datetime.datetime.now())
