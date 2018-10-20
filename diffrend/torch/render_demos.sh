python splat_render_demo.py --model ../../data/cornell.obj  --fovy 20 --out_dir ./cornell_mesh --cam_dist 2 --axis .1 .1 1 --angle 10 --at 0 0 0 --nv 10 --mesh --width=128 --height=128

python splat_render_demo.py --model ../../data/sphere_halfbox.obj  --fovy 20 --out_dir ./sphere_halfbox_mesh --cam_dist 3 --axis .8 .4 1 --angle 10 --at 0 0 0 --nv 10 --mesh --width=128 --height=128

# parametrically defined model
python splat_render_demo.py --sphere-halfbox --fovy 30 --out_dir ./sphere_halfbox_demo --cam_dist 4 --axis .8 .5 1 --angle 5 --at 0 .4 0 --nv 10 --width=256 --height=256

