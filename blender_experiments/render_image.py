import bpy
import mathutils
import os
import sys
sys.path.append('/home/user1/GitHubRepos/pix2scene/blender_experiments')
import time

from render_image_functions import *
from render_image_params import opt

start = time.time()

# Fix the image size
im_scale = 2
bpy.context.scene.render.resolution_x = opt.width*im_scale
bpy.context.scene.render.resolution_y = opt.height*im_scale
bpy.context.scene.render.resolution_percentage = 100/im_scale
bpy.context.scene.render.image_settings.compression = 0

scene = bpy.data.scenes["Scene"]

# Delete all objects in scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ADD PLANES
plane_size = 2
# XY
bpy.ops.mesh.primitive_plane_add(radius=plane_size, location=(plane_size, plane_size, 0))
# YZ
bpy.ops.mesh.primitive_plane_add(radius=plane_size, location=(0, 0, 0))
bpy.ops.transform.rotate(value=-np.pi/2, axis=(0, 1, 0))
bpy.ops.transform.translate(value=(0, plane_size, plane_size))
# ZX
bpy.ops.mesh.primitive_plane_add(radius=plane_size, location=(0, 0, 0))
bpy.ops.transform.rotate(value=-np.pi/2, axis=(1, 0, 0))
bpy.ops.transform.translate(value=(plane_size, 0, plane_size))

# i = 0
i = 0

# ADD OBJECT

# Import obj
imported_object = bpy.ops.import_scene.obj(filepath=opt.obj_filepath)

# Select the object
# obj_object = bpy.context.selected_objects[0]
for key in bpy.data.objects.keys():
    if 'Camera' not in key and 'Point' not in key:
        break

obj_key = key
obj = bpy.data.objects[obj_key]
obj.select = True

# Find objects centre
obj_center = find_center(obj)

# Translate obj to origin
bpy.ops.transform.translate(value=-obj_center)

# Flip it wrt X
# bpy.ops.transform.rotate(value=np.pi, axis=(1, 0, 0))
rotate_object(obj, np.pi, (0, 1, 0), (0, 0, 0))

# Find its max dimension
# max_dimension = max(bpy.context.selected_objects[0].dimensions)
max_dim = max(obj.dimensions)

# Resize obj with a scale
bpy.ops.transform.resize(value=(opt.new_max_dim/max_dim, opt.new_max_dim/max_dim, opt.new_max_dim/max_dim))
# Recenter it
bpy.ops.transform.translate(value=-find_center(obj))

# Rotate obj randomly about its vertical axis
rot_angles = np.random.uniform(size=opt.batch_size)*2*np.pi
# bpy.ops.transform.rotate(value=rot_angle[i], axis=(0, 0, 1))
# # Recenter it
# bpy.ops.transform.translate(value=-find_center(obj))
rotate_object(obj, rot_angles[i], (0, 0, 1), find_center(obj))

# Translate obj to awesome location
bpy.ops.transform.translate(value=(opt.new_max_dim/2, opt.new_max_dim/2, opt.new_max_dim/2))

# Don't do bpy.context.selected_objects[0].location = (opt.new_max_dim/2, opt.new_max_dim/2, opt.new_max_dim/2)
# since obj.location could be different from find_center(obj)

# LAMPS

# Generate batch_size # of light positions and colors
light_eps = 0.15
# light_pos1 = np.random.rand(opt.batch_size, 3)*opt.cam_dist + light_eps
# light_pos2 = np.random.rand(opt.batch_size, 3)*opt.cam_dist + light_eps
# light_pos3 = (3, 3, 10)
light_pos = []
light_color = []
for b in range(opt.batch_size):
    light_pos_l = []
    light_color_l = []
    for l in range(opt.n_lights):
        # Position
        if opt.light_pos[l] is None:
            light_pos_l.append(tuple(np.random.rand(3)*opt.cam_dist + light_eps))
        else:
            light_pos_l.append(opt.light_pos[l])
        # Color
        if opt.light_color[l] is None:
            light_color_l.append((0.8, 0.8, 0.8))
        else:
            light_color_l.append(opt.light_color[l])
    # Append
    light_pos.append(light_pos_l)
    light_color.append(light_color_l)

# Add lamps
lights = []
for l in range(opt.n_lights):
    bpy.ops.object.lamp_add(type='POINT', location=light_pos[i][l])
    bpy.context.object.data.color = light_color[i][l]
    lights.append(bpy.context.selected_objects[0])
    # bpy.ops.object.lamp_add(type='AREA', radius=0.5, location=(0, 0, 10))

# CAMERA

# Generate batch_size # of camera positions
cam_pos = uniform_sample_sphere(radius=opt.cam_dist, num_samples=opt.batch_size,
                                theta_range=np.deg2rad(opt.theta_range), phi_range=np.deg2rad(opt.phi_range))

# Add camera
bpy.ops.object.camera_add(location=cam_pos[i], rotation=(0, 0, 0))
camera = bpy.data.objects['Camera']

# Set field of view
bpy.context.object.data.angle = np.deg2rad(opt.fovy)

# Make this camera the camera of the scene
bpy.data.scenes[0].camera = camera

# LOOK AT
# look_at = mathutils.Vector(opt.lookat)
make_cam_lookat(camera, find_center(obj))

# RENDER
# Render settings
bpy.data.scenes[0].render.image_settings.file_format = 'PNG'
bpy.data.scenes[0].render.filepath = os.path.join(opt.save_dir, "blender_image_{0:05d}".format(i))
# Don't render shadows
bpy.context.scene.render.use_shadows = False
# Render
if opt.save_image:
    bpy.ops.render.render(write_still=True)

# RENDER DEPTH IMAGE
render_depth_image(camera, os.path.join(opt.save_dir, "blender_image_{0:05d}_depth".format(i)))

# If batch_size > 1:
for i in range(1, opt.batch_size):
    # Randomly rotate object
    obj_random_rot(obj, rot_angles[i])
    # Translate camera
    bpy.data.scenes[0].camera.location = cam_pos[i]
    # Make it look at lookat
    # looking_direction = camera.location - look_at
    # rot_quat = looking_direction.to_track_quat('Z', 'Y')
    # camera.rotation_euler = rot_quat.to_euler()
    make_cam_lookat(camera, find_center(obj))
    # Translate light
    for l in range(opt.n_lights):
        if opt.light_pos[l] is None:
            lights[l].location = light_pos[i][l]
    # Render image
    bpy.data.scenes[0].render.filepath = os.path.join(opt.save_dir, "blender_image_{0:05d}".format(i))
    if opt.save_image:
        bpy.ops.render.render(write_still=True)
        # Render depth image
        render_depth_image(camera, os.path.join(opt.save_dir, "blender_image_{0:05d}_depth".format(i)))

duration = time.time() - start

print("Total: {0:.02f} secs".format(duration))
print("Total per image: {0:.02f} secs".format(duration/opt.batch_size))
