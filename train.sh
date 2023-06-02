# python optim_hierarchy.py configs/optim_based/teaser.yaml
TOTAL_ECHPOS=1000
python optim.py configs/optim_based/teaser.yaml \
	--model:grid_res 32 --model:psr_sigma 2 \
	--train:input_mesh None --train:out_dir out/demo_optim/res_32 \
	--train:total_epochs ${TOTAL_ECHPOS} --train:lr_pcl 0.002

python optim.py configs/optim_based/teaser.yaml \
	--model:grid_res 64 --model:psr_sigma 2 \
	--train:input_mesh out/demo_optim/res_32/vis/mesh/1000.obj --train:out_dir out/demo_optim/res_64 \
	--train:total_epochs ${TOTAL_ECHPOS} --train:lr_pcl 0.0014

python optim.py configs/optim_based/teaser.yaml --model:grid_res 128 --model:psr_sigma 2 \
	--train:input_mesh out/demo_optim/res_64/vis/mesh/1000.obj --train:out_dir out/demo_optim/res_128 \
	--train:total_epochs ${TOTAL_ECHPOS} --train:lr_pcl 0.00098

