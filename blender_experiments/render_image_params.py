# FROM parameters_halfbox_shapenet
class opt:
    batch_size = 10 # Number of views to generate
    # Image
    width=128
    height = 128
    # Object
    new_max_dim = 2
    # Camera
    cam_dist = 8
    angle = 30 # camera angle (don't need b/c lookat!)
    fovy = 26 # Field of view in the vertical direction.
    focal_length = 0.1
    theta_range = [20, 80]
    phi_range = [20, 70]
    axis = [0., 1., 0.]
    lookat = [0., 0., 0.]
    # Lights
    n_lights = 3
    light_pos = [None, (0.6*cam_dist, 0.8*cam_dist, 0.2*cam_dist), (0.8*cam_dist, 0.6*cam_dist, 0.8*cam_dist)]
    light_color = [(0.8, 0.8, 0.8), (0.8, 0.1, 0.1), (0.2, 0.8, 0.2)]
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
    save_dir = "/home/user1/blender_experiments"
