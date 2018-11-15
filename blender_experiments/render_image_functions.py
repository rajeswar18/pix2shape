import bpy
import mathutils
import numpy as np

from render_image_params import opt


def rotate_object(obj, angle, direction, point):
    # obj - a Blender object
    # angle - angle in radians
    # direction - axis to rotate around (a vector from the origin)
    # point - point to rotate around (a vector)
    R = mathutils.Matrix.Rotation(angle, 4, direction)
    T = mathutils.Matrix.Translation(point)
    M = T * R * T.inverted()
    obj.location = M * obj.location
    obj.rotation_euler.rotate(M)


def find_center(obj):
    vcos = [ obj.matrix_world * v.co for v in obj.data.vertices ]
    findCenter = lambda l: ( max(l) + min(l) ) / 2
    x,y,z  = [ [ v[i] for v in vcos ] for i in range(3) ]
    return mathutils.Vector([ findCenter(axis) for axis in [x,y,z] ])


def obj_random_rot(obj, rot_angle=None):
    # # Deselect all objects
    # bpy.ops.object.select_all(action='DESELECT')
    # # Select our object
    # obj.select = True
    # # Find its center
    # obj_center = mathutils.Vector(find_center(obj))
    # # Translate to origin
    # bpy.ops.transform.translate(value=-obj_center)
    # # Rotate about Z axis by random value
    # if rot_angle is None:
    #     rot_angle = np.random.uniform()*2*np.pi
    # bpy.ops.transform.rotate(value=rot_angle, axis=(0, 0, 1))
    # # Translate back to good place
    # # bpy.ops.transform.translate(value=(opt.new_max_dim/2, opt.new_max_dim/2, opt.new_max_dim/2))
    # bpy.ops.transform.translate(value=obj_center)
    if rot_angle is None:
        rot_angle = np.random.uniform()*2*np.pi
    rotate_object(obj, rot_angle, (0, 0, 1), find_center(obj))


def make_cam_lookat(camera, look_at):
    looking_direction = camera.location - look_at
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def make_depth_texture(obj):
    # https://blendtut.wordpress.com/2011/03/09/how-to-make-a-depth-map/
    # Render depth
    # bpy.ops.material.new()
    # bpy.ops.texture.new()
    # bpy.data.textures["Texture"].type = 'BLEND'
    # bpy.context.object.active_material.texture_slots[0].texture_coords = 'GLOBAL'
    # bpy.context.object.active_material.texture_slots[0].mapping_x = 'Z'
    # bpy.context.object.active_material.texture_slots[0].mapping_y = 'Z'
    # bpy.context.object.active_material.texture_slots[0].use_map_color_diffuse = False
    # bpy.context.object.active_material.texture_slots[0].use_map_emit = True
    # bpy.context.object.active_material.texture_slots[0].color = (1, 1, 1)
    mat = bpy.data.materials.get("DepthMaterial")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="DepthMaterial")
        tex = bpy.data.textures.new("DepthTexture", 'BLEND')
        slot = mat.texture_slots.add()
        slot.texture = tex
        slot.texture_coords = 'GLOBAL'
        slot.mapping_x = 'Z'
        slot.mapping_y = 'Z'
        slot.use_map_color_diffuse = False
        slot.use_map_emit = True
        slot.color = (1, 1, 1)
    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = mat
    else:
        # no slots
        obj.data.materials.append(mat)


def remove_depth_texture(obj):
    obj.data.materials.pop(-1, update_data=True)


# RENDER DEPTH IMAGE
def render_depth_image(camera, name):
    # TX s.t. CAMERA FACES Z
    curr_cam_loc = mathutils.Vector((camera.location))
    z_axis = mathutils.Vector((0, 0, 1))
    curr_dir = -curr_cam_loc
    rot_angle = curr_dir.angle(z_axis)
    rot_axis = (-curr_cam_loc).cross(z_axis)
    # Translate all s.t. camera is at origin
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.transform.translate(value=-curr_cam_loc)
    # Rotate all s.t. camera faces z-axis
    for key in bpy.data.objects.keys():
        curr_obj = bpy.data.objects[key]
        rotate_object(curr_obj, rot_angle, rot_axis, (0, 0, 0))
    # Translate all s.t. camera is at (0, 0, -6.5)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.transform.translate(value=(0, 0, -6.5))
    # RENDER DEPTH IMAGE
    # Make surface material as depth value,
    # Turn off all lamps
    for key in bpy.data.objects.keys():
        if 'Camera' not in key and 'Point' not in key:
            make_depth_texture(bpy.data.objects[key])
        elif 'Point' in key:
            bpy.data.objects[key].data.energy = 0
    # Render settings
    bpy.data.scenes[0].render.filepath = name
    # Render image
    if opt.save_image:
        bpy.ops.render.render(write_still=True)
    # BACKTRACK
    # Remove depth material,
    # Turn all the lamps back on
    for key in bpy.data.objects.keys():
            if 'Camera' not in key and 'Point' not in key:
                remove_depth_texture(bpy.data.objects[key])
            elif 'Point' in key:
                bpy.data.objects[key].data.energy = 1
    # Translate all s.t. camera is back at origin from (0, 0, -6.5)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.transform.translate(value=(0, 0, 6.5))
    # Rotate all s.t. camera faces original direction instead of z-axis
    for key in bpy.data.objects.keys():
        curr_obj = bpy.data.objects[key]
        rotate_object(curr_obj, -rot_angle, rot_axis, (0, 0, 0))
    # Translate all s.t. camera is at original location instead of origin
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.transform.translate(value=curr_cam_loc)


def uniform_sample_circle(radius, num_samples, normal=np.array([0., 0., 1.])):
    """Generate uniform random samples into a circle."""
    theta = np.random.rand(num_samples) * 2 * np.pi
    return radius * np.stack((np.cos(theta), np.sin(theta),
                              np.zeros_like(theta)), axis=1)


def uniform_sample_cylinder(radius, height, num_samples,
                            normal=np.array([0., 0., 1.])):
    """Generate uniform random samples into a cilinder."""
    theta = np.random.rand(num_samples) * 2 * np.pi
    z = height * (np.random.rand(num_samples) - .5)
    return radius * np.stack((np.cos(theta), np.sin(theta), z), axis=1)


def uniform_sample_sphere_patch(radius, num_samples, theta_range, phi_range):
    """Generate uniform random samples a patch defined by theta and phi ranges
       on the surface of the sphere.
       :param theta_range: angle from the z-axis
       :param phi_range: range of angles on the xy plane from the x-axis
    """
    pts_2d = np.random.rand(num_samples, 2)
    s_range = 1 - np.cos(np.array(theta_range) / 2) ** 2
    t_range = np.array(phi_range) / (2 * np.pi)
    s = min(s_range) + pts_2d[:, 0] * (max(s_range) - min(s_range))
    t = min(t_range) + pts_2d[:, 1] * (max(t_range) - min(t_range))
    # theta is angle from the z-axis
    theta = 2 * np.arccos(np.sqrt(1 - s))
    phi = 2 * np.pi * t
    pts = np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                    np.cos(theta)), axis=1) * radius
    return pts


def uniform_sample_sphere_cone(radius, num_samples, axis, angle):
    """Generate uniform random samples a patch defined by theta and phi ranges
       on the surface of the sphere.
       :param theta_range: angle from the z-axis
       :param phi_range: range of angles on the xy plane from the x-axis
    """
    theta_range = [0, angle]
    phi_range = [0, 2 * np.pi]
    # Generate samples around the z-axis
    pts = uniform_sample_sphere_patch(radius, num_samples, theta_range=theta_range, phi_range=phi_range)
    # Transform from z-axis to the target axis
    axis = ops.normalize(axis)
    ortho_axis = np.cross([0, 0, 1], axis)
    ortho_axis_norm = ops.norm(ortho_axis)
    rot_angle = np.arccos(axis[2])
    if rot_angle > 0 and ortho_axis_norm > 0:
        pts_rot = ops.rotate_axis_angle(ortho_axis, rot_angle, pts)
    elif np.abs(rot_angle - np.pi) < 1e-12:
        pts_rot = pts
        pts_rot[..., 2] *= -1
    else:
        pts_rot = pts
    return pts_rot


def uniform_sample_full_sphere(radius, num_samples):
    """Generate uniform random samples into a sphere."""
    return uniform_sample_sphere_patch(radius, num_samples, theta_range=[0, np.pi],
                                       phi_range=[0, 2 * np.pi])


def uniform_sample_sphere(radius, num_samples, theta_range=None, phi_range=None, axis=None, angle=None):
    dispatch_table = [uniform_sample_full_sphere,
                      lambda x, y: uniform_sample_sphere_cone(x, y, axis=axis, angle=angle),
                      lambda x, y: uniform_sample_sphere_patch(x, y, theta_range=theta_range, phi_range=phi_range)]
    if axis is not None and angle is not None:
        assert theta_range is None and phi_range is None
        ver = 1
    elif theta_range is not None and phi_range is not None:
        assert axis is None and angle is None
        ver = 2
    else:
        assert axis is None and angle is None and theta_range is None and phi_range is None
        ver = 0
    return dispatch_table[ver](radius, num_samples)
