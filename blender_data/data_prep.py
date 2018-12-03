import os 
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + '/scripts/')
import urllib
from multiprocessing import Pool
import scipy.io as sio
import numpy as np 
from tqdm import tqdm
from glob import glob
import random 
import shutil
from PIL import Image
from PIL import ImageOps
import argparse
from subprocess import call

# this is the dataset for object translation, it will download the object files, convert then into numpy matricies, and overlay them onto pictures from the sun dataset 

parser = argparse.ArgumentParser(description='Dataset prep for image to 3D object super resolution')
parser.add_argument('-o','--object', default=['chair','table', 'bench'], help='List of object classes to be used downloaded and converted.', nargs='+' )
parser.add_argument('-no','--num_objects', default=20, help='number of objects to be converted', type = int)
parser.add_argument('-ni','--num_images', default=10, help='number of images to be created for each scene', type = int)
parser.add_argument('-ns','--num_scenes', default=20, help='number of scenes to be created for each object', type = int)
parser.add_argument('-name', default='experiment', help='experiemnt name', type = str )
args = parser.parse_args()



#labels for the union of the core shapenet classes and the ikea dataset classes 
labels = {'03001627' : 'chair', 
 '04379243': 'table', '02858304':'boat', 
'02691156': 'plane', '02808440': 'bathtub',  '02871439': 'bookcase', 
'02773838': 'bag', '02801938': 'basket', '02828884' : 'bench','02880940': 'bowl' , 
'02924116': 'bus', '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car', '03207941': 'dishwasher', 
'03211117' : 'display', '03337140': 'file', '03624134': 'knife', '03642806': 'laptop', '03710193': 'mailbox',
'03761084': 'microwave', '03928116': 'piano', '03938244':'pillow', '03948459': 'pistol', '04004475': 'printer', 
'04099429': 'rocket', '04256520': 'sofa', '04554684': 'washer' }
 
 

wanted_classes=[]
for l in labels: 
	if labels[l] in args.object:
		wanted_classes.append(l)



debug_mode = False # change to make all of the called scripts print their errors and warnings 
if debug_mode:
	io_redirect = ''
else:
	io_redirect = ' > /dev/null 2>&1'


# make data directories 
if not os.path.exists('data/objects/'):
	os.makedirs('data/objects/')



# download .obj obect files 
def download():
	with open('binvox_file_locations.txt','rb') as f: # location of all the binvoxes for shapenet's core classes 
		content = f.readlines()

	# make data sub-directories for each class
	for s in wanted_classes: 
		obj = 'data/objects/' + labels[s]+'/'
		if not os.path.exists(obj):
			os.makedirs(obj)

	# search object for correct object classes
	binvox_urls = []
	obj_urls = []
	for file in content: 
		current_class = file.split('/')
		if current_class[1] in wanted_classes:  
			if '_' in current_class[3]: continue 
			if 'presolid' in current_class[3]: continue 
			obj_urls.append(['http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/'+file.split('/')[1]+'/'+file.split('/')[2]+'/model.obj', 'data/objects/'+labels[current_class[1]]+ '/'+ current_class[2]+'.obj'])
	
	# get randomized sample from each object class of correct size
	random.shuffle(obj_urls)
	final_urls = []
	dictionary = {}
	for o in obj_urls:
		obj_class = o[1].split('/')[-2]
		if obj_class in dictionary: 
			dictionary[obj_class] += 1
			if dictionary[obj_class]> args.num_objects: 
				continue
		else: 
			dictionary[obj_class] = 1
		final_urls.append(o) 
	
	# parallel downloading of object .obj files
	commands = []
	for f in tqdm(final_urls): 
		commands.append(f)
		if len(commands) == 50: 
			pool = Pool()
			pool.map(down, commands)
			commands = []



	pool = Pool()
	pool.map(down, commands)


# download .mtl files for each .obj file to add textures during image processing 
def process_mtl(): 
	import requests
	from bs4 import BeautifulSoup
	location = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/'
	commands = []
	for s in tqdm(wanted_classes): 
		files = glob('data/objects/' + labels[s]+'/*.obj')
		
		for f in files: 
			file = f.split('/')[-1][:-4]
			if not os.path.exists('data/objects/' + labels[s]+'/' + file + '/images/'):
				os.makedirs('data/objects/' + labels[s]+'/' + file + '/images/')
			if not os.path.exists('data/objects/' + labels[s]+'/' + file +  '/' + file + '/'):
				os.makedirs('data/objects/' + labels[s]+'/' + file +  '/' + file + '/')



			shutil.move(f,'data/objects/' + labels[s]+'/' + file + '/' + f.split('/')[-1])
			commands.append([location+s+'/'+file+'/model.mtl', 'data/objects/' + labels[s]+'/' + file + '/model.mtl'])
			
			soup = BeautifulSoup(requests.get(location+s+'/'+file+'/images/').text,  "html5lib")
			for a in soup.find_all('a', href=True):
				if 'jpg' in a['href'] or 'png' in a['href']: 
					commands.append([location+s+'/'+file+'/images/'+a['href'], 'data/objects/' + labels[s]+'/' + file + '/images/'+ a['href'] ])

			soup = BeautifulSoup(requests.get(location+s+'/'+file+ '/' + file + '/').text,  "html5lib")
			for a in soup.find_all('a', href=True):
				if 'jpg' in a['href']  or 'png' in a['href']: 
					commands.append([location+s+'/'+file+  '/' + file + '/'+a['href'], 'data/objects/' + labels[s]+'/' + file + '/'+ file +'/'+ a['href'] ])

			if len(commands) >= 50: 
				pool = Pool()
				pool.map(down, commands)
				commands = []

	pool = Pool()
	pool.map(down, commands)



# these are two simple fucntions for parallel processing, down() downloads , and call() calls functions 
def down(url):
	urllib.urlretrieve(url[0], url[1])
def call(command):
	os.system('%s %s' % (command, io_redirect))



# splits each object classes into training, validation and test set in ration 70:10:20
def split():
	for s in wanted_classes: 
		dirs = glob('data/objects/' + labels[s]+'/*')
		dirs = [d for d in dirs if ( 'train' not in d) and ('test' not in d) and ('valid' not in d )]
		random.shuffle(dirs)
		train = dirs[:int(len(dirs)*.7)]
		valid = dirs[int(len(dirs)*.7):int(len(dirs)*.8)]
		test  = dirs[int(len(dirs)*.8):]
		if not os.path.exists('data/objects/' + labels[s]+'/train/'):
			os.makedirs('data/objects/' + labels[s]+'/train/')
		if not os.path.exists('data/objects/' + labels[s]+'/valid/'):
			os.makedirs('data/objects/' + labels[s]+'/valid/')
		if not os.path.exists('data/objects/' + labels[s]+'/test/'):
			os.makedirs('data/objects/' + labels[s]+'/test/')
		for t in train: 
			shutil.move(t , 'data/objects/' + labels[s]+'/train/' + t.split('/')[-1])
		for t in valid: 
			shutil.move(t , 'data/objects/' + labels[s]+'/valid/' + t.split('/')[-1])
		for t in test: 
			shutil.move(t , 'data/objects/' + labels[s]+'/test/' + t.split('/')[-1])
		


 # code for rendering the cad models in  images 
def render():
	
	sets = ['train', 'valid', 'test']
	for place in sets:
		print '------------' 
		print 'doing: ' + place
		print '------------'

		

		img_dir = 'data/images/'+ args.name + '/'  + place + '/'
		if not os.path.exists(img_dir):
			os.makedirs(img_dir)
		models = []

		for o in args.object: 
			model_dir = 'data/objects/' + o + '/' + place
			models += glob(model_dir+'/*/*.obj')
		l=0
		commands = []

		for i in tqdm(range(args.num_scenes)): 
			model_names = [random.choice(models) for k in range(2)]
			# print model_names

			target = img_dir + str(i) +'/'
			if not os.path.exists(target):
				os.mkdir(target)

			python_cmd = 'blender blender2.blend -b -P blend.py -- %s %s %s  %s ' %(args.num_images, target,  model_names[0], model_names[1])
			commands.append(python_cmd)		
			# call(commands[0]), exit()	
			
			if l%50 == 49: 

				pool = Pool()
				pool.map(call, commands)
				pool.close()
				pool.join()
				commands = []
				
			l+=1
		pool = Pool()
		pool.map(call, commands)
		pool.close()
		pool.join()
		commands = []



print '------------'
print'downloading objs'
download()
print '------------'
print'downloading mlts'
process_mtl()
print '------------'
print'splitting data'
split()
print '------------'
print'rendering images'
render()
print'finished eratin'



