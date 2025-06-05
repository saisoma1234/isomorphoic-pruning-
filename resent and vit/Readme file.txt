these are the steps to be followed :
to run the model Resnet 50 firstly :
1) prune the data using this command 
!python prune.py \
  --model resnet50 \
  --is-torchvision \
  --data-path ./data \
  --train-batch-size 32 \
  --val-batch-size 128 \
  --taylor-batchs 10 \
  --pruning-type taylor \
  --global-pruning \
  --round-to 1 \
  --save-as ./pruned_resnet50.pth \
  --test-accuracy \
  --wandb-project "MyPruningExperiments" \
  --wandb-name "ResNet50_Taylor_Pruning_50_Percent" \
  --wandb-mode "online" # or "offline" to run locally without syncing immediately

2) then create a new finetune.py file and  save this file in the path of where prune file is located and copy the code of resent finetunecode and  run this command:
!python finetune.py \
  --ckpt ./pruned_resnet50.pth \
  --model-name resnet50 \
  --data-path ./data \
  --train-batch-size 32 \
  --val-batch-size 128 \
  --resize 224 \
  --epochs 3 \
  --lr 0.01 \
  --optimizer sgd \
  --momentum 0.9 \
  --wd 5e-4 \
  --lr-scheduler cosine \
  --label-smoothing 0.1 \
  --round-to 1 \
  --is-pruned \
  --trusted-source \
  --save-as ./finetuned_pruned_resnet50.pth \
  --wandb-project "MyFineTuningProject" \
  --wandb-name "Pruned_ResNet50_FT_Run1" \
  --wandb-mode "online" # Or "offline" if you want to sync later

 For VIT model 
 to prune run this command :
1)!python3 prune.py \
  --model-name deit_small_patch16_224 \
  --data-path ./data \
  --train-batch-size 32 \
  --val-batch-size 128 \
  --taylor-batchs 10 \
  --pruning-ratio 0.5 \
  --round-to 2 \
  --save-as ./pruned_vit.pth \
  --wandb-project "MyPruningProject"  
2) then create a new finetune.py file and  save this file in the path of where prune file is located and copy the code of vit  finetunecode and  run this command:
!python3 finetune.py \
  --ckpt ./pruned_vit.pth \
  --model-name deit_small_patch16_224 \
  --data-path ./data \
  --epochs 3 \
  --train-batch-size 32 \
  --val-batch-size 64 \
  --lr 0.005 \
  --optimizer adamw \
  --wd 0.05 \
  --lr-scheduler cosine \
  --save-as ./finetuned_vit_best.pth \
  --num-workers 4 \
  --wandb-project "MyFinetuneProject"
