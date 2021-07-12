# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import segmentation_models_pytorch as smp
from preprocess import main_dataset, visualize, vis_dataset
import albumentations as album

def get_training_augmentation():
    train_transform = [    
        album.RandomCrop(height=720, width=720, always_apply=True),
        
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
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
    
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='Epoch count', required=False)
    
    parser.add_argument('-i', '--image_count',type=int, default=4200,
                        help='The count of images from the dataset for training and validation'
                        , required=False)
    
    parser.add_argument('-c', '--use_cpu', action='store_true',
                        help='If this option supplied, the network will be evaluated using the cpu.'
                             ' Otherwise the gpu will be used to run evaluation.',required=False)
    
    parser.add_argument('-d', '--data_dir', default='/home/data1/Rs19_demo/',
                        help='Location of the dataset.'
                        , required=False)
    
    parser.add_argument('-v', '--val_split',type=float, default=0.3,
                        help='Validation Split for training and validation set.'
                        , required=False)
    
    parser.add_argument('-s', '--save_dir', default='./railsem_trained_weights_new_temp/',
                        help='New weights will be saved here after training.'
                        , required=False)
    
    
    args, unknown = parser.parse_known_args()
    print(args)


    if (args.use_cpu):
        DEVICE = 'cpu'
    DEVICE = 'cuda:2'

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
    org_dataset = main_dataset(
        images_dir, 
        masks_dir, 
        args.image_count,
        augmentation = get_training_augmentation(),
        preprocessing = get_preprocessing(preprocessing_fn),
        classes = CLASSES
    )


    validation_split = args.val_split
    train_dataset, val_dataset = random_split(org_dataset, [int(args.image_count*(1-args.val_split)),
                                                          int(args.image_count*args.val_split)] )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)#, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)#, num_workers=4)

    

    TRAIN = True

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

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    val_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    max_score = 2

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_name = args.save_dir+'DeepLabV3Plus'+'.pth'
    save_name_best = args.save_dir+'DeepLabV3Plus'+'_bestloss.pth'

    print('Training on {}'.format(args.net))
    for i in range(0, args.epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = val_epoch.run(val_loader)
        
        # saving model ...
        if max_score > valid_logs['dice_loss']:
            max_score = valid_logs['dice_loss']

            torch.save(model, save_name_best)
            print('Model saved!')
        torch.save(model, save_name)
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

  
if __name__ == "__main__":

    main()
