from torch.utils.data import DataLoader, ConcatDataset
from dataset import GuitarDataset, cqt_feature, cqt_label
from model import GuitarFormer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import pickle
import os

# load train data
if os.path.exists('./train_data.pkl'):
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
else:
    solo_dataset = GuitarDataset(dataset_dir='./dataset/solo', fn_feature=cqt_feature, fn_label=cqt_label)
    comp_dataset = GuitarDataset(dataset_dir='./dataset/comp', fn_feature=cqt_feature, fn_label=cqt_label)
    train_data = ConcatDataset([solo_dataset, comp_dataset])
    with open('train_data.pkl', 'wb') as f:
        f.write(pickle.dumps(train_data))
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)


# load validate data 
if os.path.exists('./val_data.pkl'):
    with open('val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
else:
    val_data = GuitarDataset(dataset_dir='./dataset/test', fn_feature=cqt_feature, fn_label=cqt_label)
    with open('val_data.pkl', 'wb') as f:
        f.write(pickle.dumps(val_data))
val_loader = DataLoader(val_data, batch_size=512, shuffle=False)
       
# start train 
logger = TensorBoardLogger("LOG", name="GuitarsoloTranscriber", log_graph=True)
trainer = Trainer(logger=logger)

lr_monitor = LearningRateMonitor(logging_interval='step')   
# trainer = Trainer(gpus=[1], max_epochs=200, val_check_interval=1.0, logger=logger, callbacks=[lr_monitor])
trainer = Trainer(gpus=[1], max_epochs=200, check_val_every_n_epoch=1, logger=logger, callbacks=[lr_monitor])
model = GuitarFormer(d_feat=84, n_frame=4)
trainer.fit(model, train_loader, val_loader)


        
        
