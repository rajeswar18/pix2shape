import time
import tqdm

start = time.time()

n_times = 100

from main import *

opt = Parameters().parse()

opt.batchSize = 10
# opt.no_cuda = True
save_images = False

opt.root_dir = '/u/voletivi/datasets/diffrend/data/sculptures/einstein_10000'

opt.name = "{0:%Y%m%d_%H%M%S}_{1}_{2}".format(datetime.datetime.now(), opt.name, os.path.basename(opt.root_dir.rstrip('/')))

# Create experiment output folder
exp_dir = os.path.join(opt.out_dir, opt.name)
mkdirs(exp_dir)
sub_dirs=['vis_images','vis_xyz','vis_monitoring']
for sub_dir in sub_dirs:
    dir_path = os.path.join(exp_dir, sub_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    setattr(opt, sub_dir, dir_path)

dataset_load = Dataset_load(opt)

# Create GAN
gan = GAN(opt, dataset_load, exp_dir)

# Save images
if save_images:
    gan.iteration_no = 0
# Don't save images
else:
    gan.iteration_no = 1

for i in tqdm.tqdm(range(n_times)):
    gan.get_real_samples()

duration = time.time() - start

print("Total: {0:.02f} secs".format(duration))
print("Total per image: {0:.02f} secs".format(duration/opt.batchSize/n_times))


