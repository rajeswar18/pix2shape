from __future__ import division
import numpy as np
import torch
from diffrend.torch.utils import (tonemap, ray_object_intersections,
                                  generate_rays, where, backface_labeler,
                                  bincount, tch_var_f, norm_p, normalize,
                                  lookat, reflect_ray, estimate_surface_normals, tensor_dot)
from diffrend.utils.utils import get_param_value
from diffrend.torch.ops import perspective, inv_perspective
"""
Scalable Rendering TODO:
1. Backface culling. Cull splats for which dot((eye - pos), normal) <= 0 [DONE]
1.1. Filter out objects based on backface labeling.
2. Frustum culling
3. Ray culling: Low-res image and per-pixel frustum culling to determine the
   valid rays
4. Bound sphere for splats
5. OpenGL pass to determine visible splats. I.e. every pixel in the output
   image will have the splat index, the intersection point
6. Specialized version that does not render any non-planar geometry. For these the
normals per pixel do not need to be stored.
Implement ray_planar_object_intersection
"""

class Renderer:
    def __init__(self, **params):
        self.camera = params['camera']
        self._init_rays(self.camera)

    def _init_rays(self, camera):
        viewport = np.array(camera['viewport'])
        W, H = viewport[2] - viewport[0], viewport[3] - viewport[1]
        aspect_ratio = W / H

        x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
        n_pixels = x.size

        fovy = np.array(camera['fovy'])
        focal_length = np.array(camera['focal_length'])
        h = np.tan(fovy / 2) * 2 * focal_length
        w = h * aspect_ratio

        x *= w / 2
        y *= h / 2

        x = tch_var_f(x.ravel())
        y = tch_var_f(y.ravel())

        eye = camera['eye'][:3]
        at = camera['at'][:3]
        up = camera['up'][:3]

        proj_type = camera['proj_type']
        if proj_type == 'ortho' or proj_type == 'orthographic':
            ray_dir = normalize(at - eye)[:, np.newaxis]
            ray_orig = torch.stack((x, y, tch_var_f(np.zeros(n_pixels)), tch_var_f(np.ones(n_pixels))), dim=0)
            # inv_view_matrix = lookat_inv(eye=eye, at=at, up=up)
            # ray_orig = torch.mm(inv_view_matrix, ray_orig)
            # ray_orig = (ray_orig[:3] / ray_orig[3][np.newaxis, :]).permute(1, 0)
        elif proj_type == 'persp' or proj_type == 'perspective':
            ray_orig = eye[np.newaxis, :]
            ray_dir = torch.stack((x, y, tch_var_f(-np.ones(n_pixels) * focal_length)), dim=0)
            # inv_view_matrix = lookat_rot_inv(eye=eye, at=at, up=up)
            # ray_dir = torch.mm(inv_view_matrix, ray_dir)

            # normalize ray direction
            ray_dir /= torch.sqrt(torch.sum(ray_dir ** 2, dim=0))
        else:
            raise ValueError("Invalid projection type")

        self.ray_orig = ray_orig
        self.ray_dir = ray_dir
        self.H = H
        self.W = W
        return ray_orig, ray_dir, H, W

    def render(self, scene, **params):
        pass


def fragment_shader(frag_normals, light_dir, cam_dir,
                    light_attenuation_coeffs, frag_coeffs,
                    light_colors, ambient_light,
                    frag_albedo, double_sided,
                    use_quartic, light_visibility):
    # Fragment shading
    #light_dir = light_pos[:, np.newaxis, :] - frag_pos
    light_dir_norm = torch.sqrt(torch.sum(light_dir ** 2, dim=-1))[:, :, np.newaxis]
    light_dir /= light_dir_norm  # TODO: nonzero_divide
    # Attenuate the lights
    pow_val = 2 if not use_quartic else 4
    per_frag_att_factor = 1 / (light_attenuation_coeffs[:, 0][:, np.newaxis, np.newaxis] +
                               light_dir_norm * light_attenuation_coeffs[:, 1][:, np.newaxis, np.newaxis] +
                               (light_dir_norm ** pow_val) * light_attenuation_coeffs[:, 2][:, np.newaxis, np.newaxis])

    # Diffuse component
    frag_normal_dot_light = tensor_dot(frag_normals, per_frag_att_factor * light_dir, axis=-1)

    # Specular component
    reflected_light_dir = reflect_ray(-light_dir, frag_normals)

    # View dir, i.e., from vector originating from frag_pos towards the eye
    #cam_dir = normalize(camera['eye'][np.newaxis, np.newaxis, :3] - frag_pos[:, :, :3])
    reflected_cam_dot = tensor_dot(cam_dir, reflected_light_dir, axis=-1)

    if double_sided:
        # Flip per-fragment normals if needed based on the camera direction
        dot_prod = tensor_dot(cam_dir, frag_normals, axis=-1)
        sgn = torch.sign(dot_prod)
        frag_normal_dot_light = sgn * frag_normal_dot_light
        reflected_cam_dot = sgn * reflected_cam_dot

    frag_normal_dot_light = torch.nn.functional.relu(frag_normal_dot_light)
    reflected_cam_dot = torch.nn.functional.relu(reflected_cam_dot)

    ambient_reflection = ambient_light[np.newaxis, np.newaxis, :] * frag_albedo[np.newaxis, :, :]
    light_albedo_vis = light_colors[:, np.newaxis, :] * frag_albedo[np.newaxis, :, :]
    if light_visibility is not None:
        light_albedo_vis = light_albedo_vis * light_visibility[:, :, np.newaxis]
    im_color = (frag_coeffs[:, 0][np.newaxis, :, np.newaxis] * frag_normal_dot_light[:, :, np.newaxis] +
                frag_coeffs[:, 1][np.newaxis, :, np.newaxis] *
                (reflected_cam_dot[:, :, np.newaxis] ** frag_coeffs[:, 2][np.newaxis, :, np.newaxis])) * \
               light_albedo_vis + ambient_reflection
    return im_color


def render(scene, **params):
    """Render.

    :param scene: Scene description
    :return: [H, W, 3] image
    """
    # Construct rays from the camera's eye position through the screen
    # coordinates
    camera = scene['camera']
    ray_orig, ray_dir, H, W = generate_rays(camera)
    H = int(H)
    W = int(W)
    num_pixels = H * W

    scene_objects = scene['objects']

    if get_param_value('backface_culling', params, False):
        # Add a binary label per planar geometry.
        # 1: Facing away from the camera, i.e., back-face, i.e., dot(camera_dir, normal) < 0
        # 0: Facing the camera.
        # Labels are stored in the key 'backface'
        # Note that doing this before ray object intersection test reduces memory but may not result in correct
        # rendering, e.g, when an object is occluded by a back-face.
        scene_objects = backface_labeler(ray_orig, scene_objects)

    # Ray-object intersections
    disable_normals = get_param_value('norm_depth_image_only', params, False)
    if get_param_value('tiled', params, True):
        im_depth_all = []
        nearest_obj_all = []
        frag_normals_all = []
        frag_pos_all = []
        tile_size = get_param_value('tile_size', params, 4096)
        n_partitions = int(np.ceil(num_pixels / tile_size))
        for idx in range(n_partitions):
            start_idx = idx * tile_size
            end_idx = min((idx + 1) * tile_size, num_pixels)
            ray_dir_subset = ray_dir[:, start_idx:end_idx]
            obj_intersections, ray_dist, normals, material_idx = ray_object_intersections(ray_orig, ray_dir_subset,
                                                                                          scene_objects,
                                                                                          disable_normals=disable_normals)
            # num_objects = obj_intersections.size()[0]
            # Valid distances
            valid_pixels = (camera['near'] <= ray_dist) * (ray_dist <= camera['far'])
            pixel_dist = where(valid_pixels, ray_dist, camera['far'] + 1)

            # Nearest object depth and index
            im_depth, nearest_obj = pixel_dist.min(0)

            frag_normals = torch.gather(
                normals, 0, nearest_obj[np.newaxis, :, np.newaxis].repeat(1, 1, 3))
            frag_pos = torch.gather(
                obj_intersections, 0,
                nearest_obj[np.newaxis, :, np.newaxis].repeat(1, 1, 3))

            im_depth_all.append(im_depth)
            nearest_obj_all.append(nearest_obj)
            frag_normals_all.append(frag_normals)
            frag_pos_all.append(frag_pos)

            del obj_intersections
            del valid_pixels
            del pixel_dist
        im_depth = torch.cat(im_depth_all)
        nearest_obj = torch.cat(nearest_obj_all)
        frag_pos = torch.cat(frag_pos_all, dim=1)
        frag_normals = torch.cat(frag_normals_all, dim=1)
    else:
        obj_intersections, ray_dist, normals, material_idx = ray_object_intersections(ray_orig, ray_dir, scene_objects,
                                                                                      disable_normals=disable_normals)
        #num_objects = obj_intersections.size()[0]
        # Valid distances
        valid_pixels = (camera['near'] <= ray_dist) * (ray_dist <= camera['far'])
        pixel_dist = where(valid_pixels, ray_dist, camera['far'] + 1)

        # Nearest object depth and index
        im_depth, nearest_obj = pixel_dist.min(0)

        frag_normals = torch.gather(
            normals, 0, nearest_obj[np.newaxis, :, np.newaxis].repeat(1, 1, 3))
        frag_pos = torch.gather(
            obj_intersections, 0,
            nearest_obj[np.newaxis, :, np.newaxis].repeat(1, 1, 3))

        del valid_pixels
        del pixel_dist
        del obj_intersections
        del normals
        valid_pixels = None
        pixel_dist = None

    # Reshape to image for visualization
    # use nearest_obj for gather/select the pixel color
    # im_depth = torch.gather(pixel_dist, 0, nearest_obj[np.newaxis, :]).view(H, W)
    im_depth = im_depth.view(H, W)

    # Find the number of pixels covered by each object
    if get_param_value('vis_stat', params, False):
        raise RuntimeError('Removed Support for vis_stat')
        pixel_obj_count = torch.sum(valid_pixels, dim=0)
        valid_pixels_mask = pixel_obj_count > 0
        nearest_obj_only = torch.masked_select(nearest_obj, valid_pixels_mask)
        obj_pixel_count = bincount(nearest_obj_only, num_objects)
        valid_pixels_mask = valid_pixels_mask.view(H, W)
    else:
        obj_pixel_count = None
        pixel_obj_count = None
        valid_pixels_mask = None

    if get_param_value('norm_depth_image_only', params, False):
        min_depth = torch.min(im_depth)
        norm_depth_image = where(im_depth >= camera['far'], min_depth, im_depth)
        norm_depth_image = (norm_depth_image - min_depth) / (torch.max(im_depth) - min_depth)
        return {
            'image': norm_depth_image,
            'depth': im_depth,
            'ray_dist': ray_dist,
            'obj_dist': pixel_dist,
            'nearest': nearest_obj.view(H, W),
            'ray_dir': ray_dir,
            'valid_pixels': valid_pixels,
            'obj_pixel_count': obj_pixel_count,
            'pixel_obj_count': pixel_obj_count,
            'valid_pixels_mask': valid_pixels_mask,
        }

    ##############################
    # Fragment processing
    ##############################
    # Lighting
    color_table = scene['colors']
    light_pos = scene['lights']['pos'][:, :3]
    light_clr_idx = scene['lights']['color_idx']
    light_colors = color_table[light_clr_idx]
    light_attenuation_coeffs = scene['lights']['attenuation']
    #if 'ambient' not in scene['lights']:
    #    ambient_light = tch_var_f([0.0, 0.0, 0.0])
    #else:
    ambient_light = scene['lights']['ambient']

    material_albedo = scene['materials']['albedo']
    material_coeffs = scene['materials']['coeffs']

    # Generate the fragments
    """
    Get the normal and material for the visible objects.
    """

    tmp_idx = torch.gather(material_idx, 0, nearest_obj)
    frag_albedo = torch.index_select(material_albedo, 0, tmp_idx)
    frag_coeffs = torch.index_select(material_coeffs, 0, tmp_idx)

    # TODO: SOFT light visibility from fragment position
    # Generate rays from fragment position towards the light sources
    num_lights = light_pos.shape[0]
    if get_param_value('shadow', params, False):
        light_visibility = []
        for idx in range(num_lights):
            frag_to_light_dir = (light_pos[idx, np.newaxis, :] - frag_pos).squeeze().transpose(1, 0)
            frag_to_light_dist = norm_p(frag_to_light_dir.transpose(1, 0), 2)
            frag_to_light_dir /= frag_to_light_dist
            frag_ray_orig = frag_pos.squeeze() + 0.1 * frag_to_light_dir.transpose(1, 0)
            tile_size = get_param_value('tile_size', params, 4096)
            n_partitions = int(np.ceil(num_pixels / tile_size))
            single_light_vis = []
            for idx in range(n_partitions):
                start_idx = idx * tile_size
                end_idx = min((idx + 1) * tile_size, num_pixels)
                ray_dir_subset = frag_to_light_dir[:, start_idx:end_idx]
                frag_ray_orig_subset = frag_ray_orig[start_idx:end_idx, :]
                _, ray_dist, _, _ = ray_object_intersections(frag_ray_orig_subset, ray_dir_subset,
                                                             scene_objects, disable_normals=True)
                valid_dist = (ray_dist > 0) * (ray_dist < frag_to_light_dist[start_idx:end_idx])
                ray_dist = where(valid_dist, ray_dist, 1001)
                nearest_depth, nobj_idx = ray_dist.min(0)
                b_light_visible = (((nearest_depth == 1001) + (nobj_idx == nearest_obj[start_idx:end_idx])) > 0).type(torch.cuda.FloatTensor)
                single_light_vis.append(b_light_visible)
            light_visibility.append(torch.cat(single_light_vis))
        light_visibility = torch.stack(light_visibility, dim=0)
    else:
        light_visibility = None  # tch_var_f(np.ones((num_lights, H * W)))

    im_color = fragment_shader(frag_normals=frag_normals,
                               light_dir=light_pos[:, np.newaxis, :] - frag_pos,
                               cam_dir=normalize(camera['eye'][np.newaxis, np.newaxis, :3] - frag_pos[:, :, :3]),
                               light_attenuation_coeffs=light_attenuation_coeffs,
                               frag_coeffs=frag_coeffs,
                               light_colors=light_colors,
                               ambient_light=ambient_light,
                               frag_albedo=frag_albedo,
                               double_sided=get_param_value('double_sided', params, False),
                               use_quartic=get_param_value('use_quartic', params, False),
                               light_visibility=light_visibility)

    im = torch.sum(im_color, dim=0).view(int(H), int(W), 3)

    valid_pixels = (camera['near'] <= im_depth) * (im_depth <= camera['far'])
    im = valid_pixels[:, :, np.newaxis].float() * im

    # clip non-negative
    im = torch.nn.functional.relu(im)

    # Tonemapping
    if 'tonemap' in scene:
        im = tonemap(im, **scene['tonemap'])

    return {
        'image': im,
        'depth': im_depth,
        'normal': frag_normals.view(H, W, 3),
        'pos': frag_pos.view(H, W, 3),
        'ray_dist': ray_dist,
        #'obj_dist': pixel_dist,
        'nearest': nearest_obj.view(H, W),
        'ray_dir': ray_dir,
        #'valid_pixels': valid_pixels,
        #'obj_pixel_count': obj_pixel_count,
        #'pixel_obj_count': pixel_obj_count,
        #'valid_pixels_mask': valid_pixels_mask,
    }


def render_splats_NDC(scene, **params):
    """Render splats specified in the camera's normalized coordinate system

    For now, assume number of splats to be the number of pixels This would be relaxed later to allow subpixel rendering.
    :param scene: Scene description
    :return: [H, W, 3] image
    """
    camera = scene['camera']
    viewport = np.array(camera['viewport'])
    W, H = int(viewport[2] - viewport[0]), int(viewport[3] - viewport[1])
    aspect_ratio = W / H
    fovy = camera['fovy']
    near = camera['near']
    far = camera['far']
    eye = camera['eye'][:3]
    at = camera['at'][:3]
    up = camera['up'][:3]
    Mcam = lookat(eye=eye, at=at, up=up)
    #M = perspective(fovy, aspect_ratio, near, far)
    Minv = inv_perspective(fovy, aspect_ratio, near, far)

    splats = scene['objects']['disk']
    pos_NDC = splats['pos']
    normals_SLC = splats['normal']
    num_objects = pos_NDC.size()[0]

    # Transform params to the Camera's view frustum
    if pos_NDC.size()[-1] == 3:
        pos_NDC = torch.cat((pos_NDC, tch_var_f(np.ones((num_objects, 1)))), dim=1)
    pos_CC = torch.matmul(pos_NDC, Minv.transpose(1, 0))
    pos_CC = pos_CC / pos_CC[..., 3][:, np.newaxis]

    im_depth = norm_p(pos_CC[..., :3]).view(H, W)

    if get_param_value('norm_depth_image_only', params, False):
        min_depth = torch.min(im_depth)
        norm_depth_image = where(im_depth >= camera['far'], min_depth, im_depth)
        norm_depth_image = (norm_depth_image - min_depth) / (torch.max(im_depth) - min_depth)
        return {
            'image': norm_depth_image,
            'depth': im_depth,
            'pos': pos_CC,
            'normal': normals_SLC
        }
    ##############################
    # Fragment processing
    # -------------------
    # We can either perform the operations in the world coordinate or in the camera coordinate
    # Since the inputs are in NDC and converted to CC, converting to world coordinate would require more operations.
    # There are fewer lights than splats, so converting light positions and directions to CC is more efficient.
    ##############################
    # Lighting
    color_table = scene['colors']
    light_pos = scene['lights']['pos']
    light_clr_idx = scene['lights']['color_idx']
    light_colors = color_table[light_clr_idx]
    light_attenuation_coeffs = scene['lights']['attenuation']
    ambient_light = scene['lights']['ambient']

    material_albedo = scene['materials']['albedo']
    material_coeffs = scene['materials']['coeffs']
    material_idx = scene['objects']['disk']['material_idx']

    light_pos_CC = torch.mm(light_pos, Mcam.transpose(1, 0))

    # Generate the fragments
    """
    Get the normal and material for the visible objects.
    """
    normals_CC = normals_SLC   # TODO: Transform to CC, or assume SLC is CC
    frag_normals = normals_CC[:, :3]
    frag_pos = pos_CC[:, :3]

    frag_albedo = torch.index_select(material_albedo, 0, material_idx)
    frag_coeffs = torch.index_select(material_coeffs, 0, material_idx)
    light_visibility = None
    # TODO: CHECK fragment_shader call
    im_color = fragment_shader(frag_normals=frag_normals,
                               light_dir=light_pos_CC[:, np.newaxis, :3] - frag_pos[:, :3],
                               cam_dir=-frag_pos[:, :3],
                               light_attenuation_coeffs=light_attenuation_coeffs,
                               frag_coeffs=frag_coeffs,
                               light_colors=light_colors,
                               ambient_light=ambient_light,
                               frag_albedo=frag_albedo,
                               double_sided=get_param_value('double_sided', params, False),
                               use_quartic=get_param_value('use_quartic', params, False),
                               light_visibility=light_visibility)
    # # Fragment shading
    # light_dir = light_pos_CC[:, np.newaxis, :3] - frag_pos[:, :3]
    # light_dir_norm = torch.sqrt(torch.sum(light_dir ** 2, dim=-1))[:, :, np.newaxis]
    # light_dir /= light_dir_norm  # TODO: nonzero_divide
    # # Attenuate the lights
    # per_frag_att_factor = 1 / (light_attenuation_coeffs[:, 0][:, np.newaxis, np.newaxis] +
    #                            light_dir_norm * light_attenuation_coeffs[:, 1][:, np.newaxis, np.newaxis] +
    #                            (light_dir_norm ** 2) * light_attenuation_coeffs[:, 2][:, np.newaxis, np.newaxis])
    #
    # frag_normal_dot_light = tensor_dot(frag_normals, per_frag_att_factor * light_dir, axis=-1)
    # frag_normal_dot_light = torch.nn.functional.relu(frag_normal_dot_light)
    # im_color = frag_normal_dot_light[:, :, np.newaxis] * \
    #            light_colors[:, np.newaxis, :] * frag_albedo[np.newaxis, :, :]

    im = torch.sum(im_color, dim=0).view(int(H), int(W), 3)

    # clip non-negative
    im = torch.nn.functional.relu(im)

    # # Tonemapping
    # if 'tonemap' in scene:
    #     im = tonemap(im, **scene['tonemap'])

    return {
        'image': im,
        'depth': im_depth,
        'pos': pos_CC[:, :3].view(H, W, 3),
        'normal': normals_CC[:, :3].view(H, W, 3)
    }


def reshape_upsampled_data(x, H, W, C, K):
    x = x.view(H, W, C, K, K)
    # In numpy transpose(0, 3, 1, 4, 2)
    x = x.transpose(3, 1).transpose(3, 2).transpose(4, 3)
    return x.contiguous().view(H * W * K * K, C)


def z_to_pcl_CC(z, camera):
    viewport = np.array(camera['viewport'])
    W, H = int(viewport[2] - viewport[0]), int(viewport[3] - viewport[1])
    aspect_ratio = W / H

    fovy = camera['fovy']
    focal_length = camera['focal_length']
    h = np.tan(fovy / 2) * 2 * focal_length
    w = h * aspect_ratio

    ##### Find (X, Y) in the Camera's view frustum
    # Force the caller to set the z coordinate with the correct sign
    Z = -torch.nn.functional.relu(-z)

    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
    x *= w / 2
    y *= h / 2

    x = tch_var_f(x.ravel())
    y = tch_var_f(y.ravel())

    X = -Z * x / focal_length
    Y = -Z * y / focal_length

    return torch.stack((X, Y, Z), dim=1)


def render_splats_along_ray(scene, **params):
    """Render splats specified in the camera's coordinate system

    For now, assume number of splats to be the number of pixels This would be relaxed later to allow subpixel rendering.
    :param scene: Scene description
    :return: [H, W, 3] image
    """
    # TODO (fmannan): reuse z_to_pcl_CC
    camera = scene['camera']
    viewport = np.array(camera['viewport'])
    W, H = int(viewport[2] - viewport[0]), int(viewport[3] - viewport[1])
    aspect_ratio = W / H
    eye = camera['eye'][:3]
    at = camera['at'][:3]
    up = camera['up'][:3]
    Mcam = lookat(eye=eye, at=at, up=up)
    #M = perspective(fovy, aspect_ratio, near, far)
    #Minv = inv_perspective(fovy, aspect_ratio, near, far)

    splats = scene['objects']['disk']
    pos_ray = splats['pos']
    normals_CC = get_param_value('normal', splats, None)
    #num_objects = pos_ray.size()[0]

    fovy = camera['fovy']
    focal_length = camera['focal_length']
    h = np.tan(fovy / 2) * 2 * focal_length
    w = h * aspect_ratio

    ##### Find (X, Y) in the Camera's view frustum
    # Force the caller to set the z coordinate with the correct sign
    if pos_ray.dim() == 1:
        Z = -torch.nn.functional.relu(-pos_ray)  # -torch.abs(pos_ray[:, 2])
    else:
        Z = -torch.nn.functional.relu(-pos_ray[:, 2]) #-torch.abs(pos_ray[:, 2])

    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
    x *= w / 2
    y *= h / 2

    x = tch_var_f(x.ravel())
    y = tch_var_f(y.ravel())
    #sgn = 1 if get_param_value('use_old_sign', params, False) else -1
    X = -Z * x / focal_length
    Y = -Z * y / focal_length

    pos_CC = torch.stack((X, Y, Z), dim=1)

    if get_param_value('orient_splats', params, False) and normals_CC is not None:
        # TODO (fmannan): Orient splats so that [0, 0, 1] maps to the camera direction
        # Peform this operation only when splat normals are generated by the caller in CC
        # This should help with splats that are at the edge of the view-frustum and the camera has
        # a large fov.
        pass

    # Estimate normals from splats/point-cloud if no normals were provided
    if normals_CC is None:
        normal_est_method = get_param_value('normal_estimation_method', params, 'plane')
        kernel_size = get_param_value('normal_estimation_kernel_size', params, 3)
        normals_CC = estimate_surface_normals(pos_CC.view(H, W, 3), kernel_size, normal_est_method)[..., :3].view(-1, 3)

    material_idx = scene['objects']['disk']['material_idx']
    light_visibility = None
    if 'light_vis' in scene['objects']['disk']:
        light_visibility = scene['objects']['disk']['light_vis']

    # Samples per pixel (supersampling)
    samples = get_param_value('samples', params, 1)
    if samples > 1:
        """There are three variables that need to be upsampled:
        1. positions, 2. normals, and 3. shadow maps (light visibility)
        The idea here is to generate an x-y grid in the original resolution, then shift that
        to find the subpixels, then find the plane parameters for the splat bounded within the pixel
        frustum (i.e., a frustum projected into the scene by a pixel), and then for each subpixel
        find the ray-plane intersection with that splat plane.

        The subpixel rays are generated by taking the mesh on the projection plane and shifting it
        by the appropriate amount to get the pixel coordinate that the ray should go through, then
        finding the position in the 3D camera space. The normal of the splat is copied to all those
        surface samples.

        n_x (x - x0) + n_y (y - y0) + n_z (z - z0) = 0
        n_x x0 + n_y y0 + n_z z0 = d0
        n_x t u_x + ... = d0
        t = d0 / dot(n, ray)
        """
        # plane parameter
        d = torch.sum(pos_CC * normals_CC[:, :3], dim=1)
        z = tch_var_f(np.ones(x.shape) * -focal_length)
        # # Test consistency
        # pos_CC_projplane = torch.stack((x, y, z), dim=1)
        # dot_ray_normal = torch.sum(pos_CC_projplane * normals_CC[:, :3], dim=1)
        # t = d / dot_ray_normal
        # pos_CC_test = t[:, np.newaxis] * pos_CC_projplane
        # diff = torch.mean(torch.abs(pos_CC_test - pos_CC))
        # print(diff)
        # # End of consistency check

        # Find ray-plane intersection for the plane bounded by the frustum
        # The width and height of the projection plane are w and h
        dx = w / (samples * W - 1)  # subpixel width
        dy = h / (samples * H - 1)  # subpixel height
        pos_CC_supersampled = []
        normals_CC_supersampled = []
        material_idx_supersampled = []
        if light_visibility is not None:
            light_visibility_supersampled = []
            light_visibility = light_visibility.transpose(1, 0)
        for c, deltax in enumerate(np.linspace(-1, 1, samples)):
            # TODO (fmannan): generalize (the div by 2) for samples > 3
            xx = x + deltax * dx / 2  # Shift by half of the subpixel size
            for r, deltay in enumerate(np.linspace(1, -1, samples)):
                yy = y + deltay * dy / 2
                # unit ray going through sub-pixels
                pos_CC_projplane = normalize(torch.stack((xx, yy, z), dim=1))
                dot_ray_normal = torch.sum(pos_CC_projplane * normals_CC[:, :3], dim=1)
                t = d / dot_ray_normal

                pos_CC_supersampled.append(t[:, np.newaxis] * pos_CC_projplane)
                normals_CC_supersampled.append(normals_CC[:, :3])
                material_idx_supersampled.append(material_idx[:, np.newaxis])
                if light_visibility is not None:
                    light_visibility_supersampled.append(light_visibility)
        pos_CC_supersampled = torch.stack(pos_CC_supersampled, dim=2)
        normals_CC_supersampled = torch.stack(normals_CC_supersampled, dim=2)
        material_idx_supersampled = torch.stack(material_idx_supersampled, dim=2)
        if light_visibility is not None:
            light_visibility_supersampled = torch.stack(light_visibility_supersampled, dim=2)

        pos_CC = reshape_upsampled_data(pos_CC_supersampled, H, W, 3, samples)
        normals_CC = reshape_upsampled_data(normals_CC_supersampled, H, W, 3, samples)
        material_idx = reshape_upsampled_data(material_idx_supersampled, H, W, 1, samples).view(-1)
        if light_visibility is not None:
            light_visibility = reshape_upsampled_data(light_visibility_supersampled, H, W, light_visibility.shape[1], samples).transpose(1, 0)
        H *= samples
        W *= samples
        ####
    im_depth = norm_p(pos_CC[..., :3]).view(H, W)

    if get_param_value('norm_depth_image_only', params, False):
        min_depth = torch.min(im_depth)
        norm_depth_image = where(im_depth >= camera['far'], min_depth, im_depth)
        norm_depth_image = (norm_depth_image - min_depth) / (torch.max(im_depth) - min_depth)
        return {
            'image': norm_depth_image,
            'depth': im_depth,
            'pos': pos_CC,
            'normal': normals_CC
        }
    ##############################
    # Fragment processing
    # -------------------
    # We can either perform the operations in the world coordinate or in the camera coordinate
    # Since the inputs are in NDC and converted to CC, converting to world coordinate would require more operations.
    # There are fewer lights than splats, so converting light positions and directions to CC is more efficient.
    ##############################
    # Lighting
    color_table = scene['colors']
    light_pos = scene['lights']['pos']
    light_clr_idx = scene['lights']['color_idx']
    light_colors = color_table[light_clr_idx]
    light_attenuation_coeffs = scene['lights']['attenuation']
    ambient_light = scene['lights']['ambient']

    material_albedo = scene['materials']['albedo']
    material_coeffs = scene['materials']['coeffs']


    light_pos_CC = torch.mm(light_pos, Mcam.transpose(1, 0))

    # Generate the fragments
    """
    Get the normal and material for the visible objects.
    """
    frag_normals = normals_CC[:, :3]
    frag_pos = pos_CC[:, :3]

    frag_albedo = torch.index_select(material_albedo, 0, material_idx)
    frag_coeffs = torch.index_select(material_coeffs, 0, material_idx)

    im_color = fragment_shader(frag_normals=frag_normals,
                               light_dir=light_pos_CC[:, np.newaxis, :3] - frag_pos[:, :3],
                               cam_dir=-normalize(frag_pos[np.newaxis, :, :3]),
                               light_attenuation_coeffs=light_attenuation_coeffs,
                               frag_coeffs=frag_coeffs,
                               light_colors=light_colors,
                               ambient_light=ambient_light,
                               frag_albedo=frag_albedo,
                               double_sided=False,
                               use_quartic=get_param_value('use_quartic', params, False),
                               light_visibility=light_visibility)

    im = torch.sum(im_color, dim=0).view(int(H), int(W), 3)

    # clip non-negative
    im = torch.nn.functional.relu(im)

    # Tonemapping
    #if 'tonemap' in scene:
    #    im = tonemap(im, **scene['tonemap'])

    return {
        'image': im,
        'depth': im_depth,
        'pos': pos_CC.view(H, W, 3),
        'normal': normals_CC.contiguous().view(H, W, 3)
    }


def test_render_splat_NDC_0():
    fovy = np.deg2rad(45)
    aspect_ratio = 1
    near = 0.1
    far = 1000
    M = perspective(fovy, aspect_ratio, near, far)
    Minv = inv_perspective(fovy, aspect_ratio, near, far)

    pos_NDC = tch_var_f([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    normals_SLC = tch_var_f([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    num_objects = pos_NDC.size()[0]

    # Transform params to the Camera's view frustum
    if pos_NDC.size()[-1] == 3:
        pos_NDC = torch.cat((pos_NDC, tch_var_f(np.ones((num_objects, 1)))), dim=1)
    pos_CC = torch.matmul(pos_NDC, Minv.transpose(1, 0))
    pos_CC = pos_CC / pos_CC[..., 3][:, np.newaxis]

    pixel_dist = norm_p(pos_CC[..., :3])
