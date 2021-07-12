# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from preprocess import main_dataset, visualize, vis_dataset
import albumentations as album

def get_test_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1920, min_width=1920, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask = to_tensor))
        
    return album.Compose(_transform)

def main():
    #start_time = time.time()
    parser = argparse.ArgumentParser(description='Training DeepLabV3Plus models with RailSem dataset')

    parser.add_argument('-n', '--net', default='DeepLabV3Plus',
                        help='DeepLabV3Plus model for training', required=False)
    
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Epoch count', required=False)
    
    parser.add_argument('-i', '--image_count',type=int, default=50,
                        help='The count of images from the dataset for testing'
                        , required=False)
    
    parser.add_argument('-c', '--use_cpu', action='store_true',
                        help='If this option supplied, the network will be evaluated using the cpu.'
                             ' Otherwise the gpu will be used to run evaluation.',required=False)
    
    parser.add_argument('-d', '--data_dir', default='/home/data1/Rs19_demo/',
                        help='Location of the dataset.'
                        , required=False)
    
    parser.add_argument('-v', '--val_split',type=float, default=0.3,
                        help='Test Split for training and test set.'
                        , required=False)
    
    parser.add_argument('-s', '--save_dir', default='./railsem_trained_weights_new_temp/',
                        help='New weights will be saved here after training.'
                        , required=False)
    
    
    args, unknown = parser.parse_known_args()
    print(args)


    if (args.use_cpu):
        DEVICE = 'cpu'
    DEVICE = 'cuda:0'

    CLASSES = ['rail-track','rail-raised']
    
    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'

    model = smp.DeepLabV3Plus(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        activation =ACTIVATION,
        classes =len(CLASSES),
    )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER,ENCODER_WEIGHTS)

    images_dir = os.path.join(args.data_dir,'jpgs')
    masks_dir = os.path.join(args.data_dir,'uint8')
    jsons_dir = os.path.join(args.data_dir,'jsons')
    test_dataset = main_dataset(
        images_dir, 
        masks_dir, 
        args.image_count,
        augmentation = get_test_augmentation(),
        preprocessing = get_preprocessing(preprocessing_fn),
        classes = CLASSES
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)#, num_workers=12)
 
    
    #loss = smp.utils.losses.CrossEntropyLoss(ignore_index=255)
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        #smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Accuracy(threshold=0.5),
    ]


    #optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam([
        {'params' : model.decoder.parameters(), 'lr': 0.001},
])

    max_score = 2

    best_model = torch.load('/home/data1/Pdy_ws/squeezenas_train-master/railsem_trained_weights_new_temp/DeepLabV3Plus.pth')

    
    test_epoch = smp.utils.train.ValidEpoch(
        model = best_model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    logs = test_epoch.run(test_loader)
    

    visual_dataset = vis_dataset(
        images_dir, 
        masks_dir, 
        classes = CLASSES
    )

    for i in range(9):
        n = np.random.choice(len(test_dataset))
        
        image ,mask= visual_dataset[n]
   
        img , gt_mask= test_dataset[n]

        gt_mask_railtrack=mask[:,:,0].squeeze()
        gt_mask_railraised=mask[:,:,1].squeeze()

        x_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)

        pr_mask_railtrack = pr_mask.cpu().squeeze()[0,:,:].numpy().round()
        pr_mask_railraised = pr_mask.cpu().squeeze()[1,:,:].numpy().round()
        
        visualize(image = image,
        ground_truth_railtrack = gt_mask_railtrack,
        ground_truth_railraised = gt_mask_railraised,
        predicted_mask_railtrack = pr_mask_railtrack,
        predicted_mask_railraised = pr_mask_railraised,
        )
        
        
     
if __name__ == "__main__":

    main()