CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/home/yang/SAM2-UNet/checkpoint/sam2_hiera_large.pt" \
--train_image_path "/home/yang/SAM2-UNet/datasets/Kvasir-SEG/images/" \
--train_mask_path "/home/yang/SAM2-UNet/datasets/Kvasir-SEG/masks/" \
--save_path "/home/yang/SAM2-UNet/save/" \
--epoch 20 \
--lr 0.001 \
--batch_size 12