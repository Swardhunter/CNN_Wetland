if __name__ == "__main__":
    from Functions import MyCall,CustomDataset,UNET_BW
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer
    from pytorch_lightning.tuner.tuning import Tuner
    import os
    from pytorch_lightning.strategies import DDPStrategy
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    #chkpointpath = rf'/home/mskenawi/Mahmoud_Saber_Kenawi/WETLAND/Code/Log_Swin_Unet/Swin/version_5/checkpoints/epoch=44-step=169155.ckpt'

    # # Load the checkpoint
    #checkpoint = torch.load(chkpointpath)
    #print(chkpointpath)


    encoder = "EncoderName"
    logger = TensorBoardLogger(f"Log{encoder}",name=encoder)
    tensorspath = rf'Path to Processed Tensors'
    dataset = CustomDataset(tensorspath)
    print (len(dataset))
    
    # # Modify the hyperparameters
    #checkpoint['hyper_parameters']['batch_size'] = 78
    #checkpoint['hyper_parameters']['dataset']= dataset
    # Save the checkpoint
    #torch.save(checkpoint, "/home/mskenawi/trial.ckpt")

    
    
    model = UNET_BW (num_classes=1, endcoder=encoder,learning_rate=0.001, dataset=dataset, batch_size=16)
    #model = SWIN_UNET_BW.load_from_checkpoint("/home/mskenawi/trial.ckpt")


    trainer = Trainer(logger=logger,
            strategy=DDPStrategy(),
            max_epochs=100,
            devices='3',
            log_every_n_steps=1,
            callbacks=[MyCall(), EarlyStopping(monitor="IoU", patience=15, mode="max")]
            
        )
    tuner = Tuner(trainer)

    lr_finder = tuner.lr_find(model)

        # #     # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    trainer.fit(model)
    trainer.test(model)