

import os
import bpy
 
bpy.context.scene.cycles.device = 'GPU'
bpy.ops.render.render(True)

import sys
import math
import random
import numpy as np
from glob import glob
import io
from contextlib import redirect_stdout
from PIL import Image
import bmesh
from mathutils import Vector



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

light_num_lowbound = 0
light_num_highbound = 6
light_dist_lowbound = 8
light_dist_highbound = 20

g_syn_light_num_lowbound = 0
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 20
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 0
g_syn_light_elevation_degree_lowbound = -10
g_syn_light_elevation_degree_highbound = 10
g_syn_light_energy_mean = 2
g_syn_light_energy_std = 2
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1


def camPosToQuaternion(cx, cy, cz):
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    axis = (-cz, 0, cx)
    angle = math.acos(cy)
    a = math.sqrt(2) / 2
    b = math.sqrt(2) / 2
    w1 = axis[0]
    w2 = axis[1]
    w3 = axis[2]
    c = math.cos(angle / 2)
    d = math.sin(angle / 2)
    q1 = a * c - b * d * w1
    q2 = b * c + a * d * w1
    q3 = a * d * w2 + b * d * w3
    q4 = -b * d * w2 + a * d * w3
    return (q1, q2, q3, q4)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)


    




num_images =  int(sys.argv[-4])
target = sys.argv[-3]
models = []

models.append(sys.argv[-2])
models.append(sys.argv[-1])



# giving cubes, and walls random colours 

for o in bpy.data.objects:
    if 'Cube' in o.name or 'Plane' in o.name:
        mat = bpy.data.materials.new('')
        mat.diffuse_color = (random.uniform(.2,.7),random.uniform(.2,.7), random.uniform(.2,.7))
        o.active_material = mat 


# set camera parameters 
view_params = [[random.uniform(270,360.) , random.uniform(10,60.0),  0, random.uniform(5,7)] for i in range(num_images)]
bpy.context.scene.render.image_settings.color_mode ='RGB'
scene = bpy.context.scene
camObj = bpy.data.objects['Camera']
joined_objs = []
positions = Vector([0,0,0])
x_dim = []
y_dim = []
o_max_dim = 0 

# joining objects together and finding the their dimensions 
for i,shape_file in enumerate(models):
    bpy.ops.import_scene.obj(filepath=shape_file )    
    obs = []
    

    # find objects 
    for ob in scene.objects:
        print (ob.name)
        if ('lamp' not in ob.name and 'Plane' not in ob.name and 'Camera' not in ob.name ) and ob.name[:3] != 'obj':
            obs.append(ob)

    for ob in scene.objects:
        if ('lamp' not in ob.name and 'Plane' not in ob.name and 'Camera' not in ob.name ) and ob.name[:3] != 'obj':
            ob.select = True
            bpy.context.scene.objects.active = ob
        else:
            ob.select = False
    # print (obs)
    print ("about to join")
    bpy.ops.object.join()
    print('joined')
    o = bpy.context.selected_objects[0]
    o.name = ('obj' + str(i))
    o.scale = (1.5,1.5,1.5)
    scene.update()
    joined_objs.append(o.name)
    

 


    local_verts = [Vector(v[:]) for v in o.bound_box]
    lowest_pt = min([(o.matrix_world * v.co)[2] for v in o.data.vertices])
    o.location[2] -= lowest_pt

    x_dim_min = min([(o.matrix_world * v.co)[0] for v in o.data.vertices])
    x_dim_max = max([(o.matrix_world * v.co)[0] for v in o.data.vertices])
    o_x_dim = x_dim_max - x_dim_min
    y_dim_min = min([(o.matrix_world * v.co)[1] for v in o.data.vertices])
    y_dim_max = max([(o.matrix_world * v.co)[1] for v in o.data.vertices])
    o_y_dim = y_dim_max - y_dim_min
    o_max_dim = max(o_x_dim, o_y_dim, o_max_dim)
    x_dim.append(o_x_dim)
    y_dim.append(o_y_dim)



looking = True

# finding a place for all the cubes in the scene 
while looking: 
    looking = False 
    centers = []
    for i in range(len(models)):
        if looking: break 
        o = bpy.data.objects['obj'+str(i)]
       

        # move objs around
        proper = False 
        count = 0 
        while not proper :
            proper = True 
            x_change = np.random.uniform(0,4)
            y_change = -np.random.uniform(0,4)
            for x, y in centers: 
                if x_change+ x_dim[i] >x and x_change- x_dim[i] <x: 
                    proper = False
                elif y_change+ y_dim[i] >y and y_change- y_dim[i] <y:
                    proper = False
            if proper: 
                centers.append([x_change, y_change])
            count+= 1 
            if count> 20: 
                looking = True 
                break 
        

height = 0

#code for moving cubes and objects 
for i in range(len(models)): 
    o = bpy.data.objects['obj'+str(i)]
    x_change, y_change = centers[i]
    o.location+= Vector([x_change + x_dim[i]/2 , y_change - y_dim[i]/2, 0])
    # move object to random point on cube 
    scene.update()

centers = np.array(centers)
center = np.sum(centers, axis = 0 )/4.




cam_locations = []
cam_rotations = []
bpy.data.scenes['Scene'].render.image_settings.file_format = 'OPEN_EXR'
bpy.context.scene.world.light_settings.use_environment_light = True
bpy.context.scene.world.light_settings.environment_energy = .2
bpy.context.scene.world.light_settings.environment_color = 'PLAIN'
bpy.data.worlds['World'].horizon_color = (1, 1, 1)
scene = bpy.context.scene 
scene.render.resolution_x = 256
scene.render.resolution_y = 256
scene.render.resolution_percentage = 100


lx = np.random.uniform(1,5)
ly = np.random.uniform(-5,-1)
lz = np.random.uniform(3,7)
lamp = bpy.data.objects['lamp'+str(2)]
pos = [lx,ly,lz]
lamp.location = pos


for e, param in enumerate(view_params):

    result = target + str(e) + '.png'
    # setting camera orientation 
    azimuth_deg = param[0]
    elevation_deg = param[1]
    theta_deg = param[2]
    rho = param[3]


      

    
    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    q = quaternionProduct(q2, q1)

    # setting camera positions 
    camObj.location[0] = cx + center[0] +.5
    camObj.location[1] = cy + center[1] -.5
    camObj.location[2] = cz + height
    camObj.rotation_mode = 'QUATERNION'
    camObj.rotation_quaternion[0] = q[0]
    camObj.rotation_quaternion[1] = q[1]
    camObj.rotation_quaternion[2] = q[2]
    camObj.rotation_quaternion[3] = q[3]
    cam_locations.append([cx + center[0] +.5, cy + center[1] -.5, cz + height])
    cam_rotations.append(q)
    
    bpy.data.scenes['Scene'].render.filepath = result
  
    bpy.ops.render.render( write_still=True )
for ob in  bpy.data.objects:
    ob.active_material_index = 0
    for i in range(len(ob.material_slots)):
        bpy.ops.object.material_slot_remove({'object': ob})


for e, param in enumerate(view_params):

    result = target + str(e) + '_nc.png'
    # setting camera orientation 


    # setting camera positions 
   
    camObj.location = cam_locations[e]
    camObj.rotation_quaternion = cam_rotations[e]
    cam_locations.append(camObj)
    cam_rotations.append(camObj.rotation_quaternion)
    
    bpy.data.scenes['Scene'].render.filepath = result
    bpy.ops.render.render( write_still=True )



