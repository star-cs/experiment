import os

config_base = {
    'root_path' : '/home/yang/experiment',
    
    'ckp_path' : 'checkpoint',
    'ckp_hiera_name' : 'sam2_hiera_large.pt',
    
    'dataset_path': 'datasets',
    'dataset_label': 'Synapse' , # Kvasir 1 Synapse 7
    'num_classes' : 8,

    # Kvasir
    'train_image_name' : 'Kvasir-SEG/images/Train/',
    'train_gt_name' : 'Kvasir-SEG/masks/Train/',
    'test_image_name' : 'Kvasir-SEG/images/Val/',
    'test_gt_name' : 'Kvasir-SEG/masks/Val/', 

    # Synapse
    'Synapse_image_name' : 'Synapse/images/',
    'Synapse_masks_name' : 'Synapse/masks/',
    'Synapse_lists' : 'Synapse/',
    
    'log_path' : 'log', 
    'save_model' : 'model_log', 
    'tensorboard_path' : 'tensorboard_log',
    'csv_path' : 'csv',

    'epoch' : 20,
    'lr' : 0.001,
    'batch_size' : 2,
    'weight_decay' : 5e-4,

    'image_size' : 512,

    'embed_dim' : 576,              # hiera，插入cnn特征层时候的通道数
    'num_patchs' : 32,              # hiera，插入cnn特征层时候的尺寸

    'adapter_type' : 'adaptor',     # adaptor fully_shared fully_unshared
    'cnn_label' : 'none',       # none resnet50 convnextv2_base segnext_base
        
}

'''
可选组合 : 
['RFB'] + 'UNet'
['RCM', 'RFB'] + 'UNet'
['RCM', 'Duck'] + 'UNet'

['RCM'] + 'EMCAD'
['None'] + 'EMCAD'
'''

config_neck = {
    # None
    # RFB(SAM2_UNet模块)                       channels 不变
    # RCM(https://arxiv.org/pdf/2405.06228)    input channels --> output channels
    # Duck Block                               input channels --> output channels

    'neck_type' : ['None'], 
}

config_decoder = {
    # UNet   默认                              # 使用这个模块，neck得加RFB，把通道数全部转成 64
    # EMCAD  https://arxiv.org/abs/2405.06880  # 使用这个模块，默认neck部分就不能改变通道数。此时，neck_type只能为None
    'decoder_type' : 'EMCAD'            
}

path_config = {
    'hiera_path' : str(os.path.join(config_base['root_path'], 
                                    config_base['ckp_path'], 
                                    config_base['ckp_hiera_name'])) ,

    'convnextv2_base_ckpt_path' : str(os.path.join(config_base['root_path'],
                                         config_base['ckp_path'],
                                         'convnextv2_base_22k_224_ema.pt')), 
                                         # convnextv2_base_1k_224_fcmae.pt      convnextv2_base_22k_224_ema.pt

    'segnext_base_ckpt_path' : str(os.path.join(config_base['root_path'],
                                         config_base['ckp_path'],
                                         'mscan_b.pth')),

    # Kvasir
    'train_image_path' : str(os.path.join(config_base['root_path'], 
                                          config_base['dataset_path'],
                                          config_base['train_image_name'])) ,
                                                
    'train_mask_path' : str(os.path.join(config_base['root_path'],
                                         config_base['dataset_path'],
                                         config_base['train_gt_name'])) ,
                                         
    'test_image_path' : str(os.path.join(config_base['root_path'],
                                         config_base['dataset_path'],
                                         config_base['test_image_name'])), 

    'test_mask_path' : str(os.path.join(config_base['root_path'],
                                         config_base['dataset_path'],
                                         config_base['test_gt_name'])) ,                    
    
    # Synapse
    'Synapse_image_path' : str(os.path.join(config_base['root_path'],
                                         config_base['dataset_path'],
                                         config_base['Synapse_image_name'])) , 
    'Synapse_masks_path' : str(os.path.join(config_base['root_path'],
                                         config_base['dataset_path'],
                                         config_base['Synapse_masks_name'])) , 
    'Synapse_lists_path' : str(os.path.join(config_base['root_path'],
                                         config_base['dataset_path'],
                                         config_base['Synapse_lists'])) , 

    'save_path' : str(os.path.join(config_base['root_path'],
                                   config_base['log_path'],
                                   config_base['save_model'])),
                                   
                                   
    'tensorboard_path' : str(os.path.join(config_base['root_path'],
                                          config_base['log_path'],
                                          config_base['tensorboard_path'])),

    'csv_path' : str(os.path.join(config_base['root_path'],
                                config_base['log_path'],
                                config_base['csv_path'])), 

    'train_version' : str(config_base['cnn_label'] + "_" + 
                    config_base['adapter_type'] + "_" + 
                    '+'.join(str(item) for item in config_neck['neck_type'])  + "_" +  
                    config_decoder['decoder_type']),  #  用于区分log文件                                       
}


