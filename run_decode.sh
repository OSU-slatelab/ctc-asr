#!/bin/bash
checkpoint_name=hc_ctc_all_H512L24DS8E48
test_path=/research/nfs_fosler_1/vishal/text/libri/test_other.csv
PTH=decodes/${checkpoint_name}_other
world_size=4
rm -vrf $PTH/*.log
rm -vrf $PTH/*.txt
#python main.py \
#OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=$world_size --master_port=33543 decode.py \
OMP_NUM_THREADS=8 torchrun --standalone --nnodes=1 --nproc_per_node=$world_size decode.py \
  --config-path="configs/" \
  --config-name="hc_ctc_all" \
  corpus='librispeech' \
  distributed.world_size=$world_size \
  paths.test_path="${test_path}" \
  paths.ckpt_path="/research/nfs_fosler_1/vishal/saved_models/${checkpoint_name}.pth.tar" \
  paths.decode_path="${PTH}"
rm -vrf $PTH/full.txt
cat ${PTH}/{0..3}.txt > "${PTH}/full.txt"
python evaluate.py --path "${PTH}/full.txt"
