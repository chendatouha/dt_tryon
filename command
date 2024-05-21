test
python test_script.py -c configs/test/viton.yaml

train
python train_script.py -c configs/train/viton.yaml

demo
python -m deploy.viton.viton_demo

train animate step2
python -m accelerate.commands.launch --main_process_port=28500 --num_processes=2 train_script.py -c configs/train/animate_step2.yaml

test animate step2
python test_script.py -c configs/test/animate_step2.yaml

train animate step1
python -m accelerate.commands.launch --num_processes=2 train_script.py -c configs/train/animate_step1.yaml

finetune sdxl
python -m accelerate.commands.launch --num_processes=2 --main_process_port=29501 train_script.py -c configs/train/sdxl_finetune.yaml