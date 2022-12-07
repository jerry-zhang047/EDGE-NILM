from nilmtk.api import API
from collections import OrderedDict
import time
import random
from nilmtk.disaggregate import Disaggregator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# Use CUDA or not
USE_CUDA = torch.cuda.is_available()

# Set learning rate
learning_rate = 1e-3

# fix the random seed to guarantee the reproducibility
random_seed = 10
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class seq2seq(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 5), 0),
            nn.Conv1d(1, 30, 10, 1),
            nn.ReLU(True),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(30, 30, 8, 1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 3), 0),
            nn.Conv1d(30, 40, 6, 1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(40, 50, 5, 1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(50, 50, 5, 1),
            nn.ReLU(True)
        )
        self.dense = nn.Sequential(
            nn.Linear(50*sequence_length, 1024),
            nn.ReLU(True),
            nn.Linear(1024, sequence_length)
        )

    def forward(self, power_seq):
        inp = self.conv(power_seq).view(-1, 50 * self.sequence_length)
        out = self.dense(inp)
        return out.view(-1, 1, self.sequence_length)

def initialize(layer):
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, val=0.0)

# test the input and output of the seq2seq module
'''
sequence_length = 256
model = seq2seq(sequence_length=sequence_length)
input = torch.ones((10, 1, 256))
print(input.shape)
output = model(input)
print(output.shape)
'''

class Seq2Seq(Disaggregator):
    def __init__(self, params):
        self.MODEL_NAME = "Seq2Seq"
        self.sequence_length = params.get("sequence_length", 599)
        self.n_epochs = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)
        self.train_patience = params.get("train_patience", 3)
        self.appliance_params = params.get("appliance_params", {})
        self.models = OrderedDict()

    def partial_fit(self, train_mains, train_appliances, pretrained=False, do_preprocessing=True):
        # calculate mean, std of appliances if not set 
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        
        # do preprocessing: including z-score normalization and data slicing
        if do_preprocessing:
            print("Doing preprocessing")
            train_mains, train_appliances = self.call_preprocessing(train_mains, train_appliances, 'train')

        train_mains = pd.concat(train_mains, axis=0).values
        train_mains = train_mains.reshape((-1, 1, self.sequence_length))

        tmp_train_appliances = []
        for app_name, app_power in train_appliances:
            app_df = pd.concat(app_power, axis=0).values
            app_df = app_df.reshape((-1, 1, self.sequence_length))
            tmp_train_appliances.append((app_name, app_df))
        train_appliances = tmp_train_appliances

        for app_name, app_power in train_appliances:
            if app_name not in self.models:
                print("First model training for ", app_name)
                self.models[app_name] = seq2seq(self.sequence_length)

            model = self.models[app_name]
            train(app_name, model, train_mains, app_power, self.n_epochs, self.batch_size, self.train_patience)
            self.models[app_name].load_state_dict(torch.load("./"+app_name+"_seq2seq_best_state_dict.pt"))

    def disaggregate_chunk(self, test_mains, do_preprocessing=True):
        if do_preprocessing:
            test_mains = self.call_preprocessing(test_mains, submeters_lst=None, method='test')

        test_pred = []
        for test_main in test_mains:
            test_main = test_main.values.reshape((-1, 1, self.sequence_length))
            disaggregate_dict = {}
            for appliance in self.models:
                model = self.models[appliance].to('cpu')
                pred = test(model, test_main, self.batch_size)
                mean, std = self.appliance_params[appliance]['mean'], self.appliance_params[appliance]['std']
                pred = self.denormalize_data(pred, mean, std)
                # filter valid predictions by zero-assignment
                valid_pred = pred.flatten()
                valid_pred = np.where(valid_pred > 0, valid_pred, 0)
                disaggregate_dict[appliance] = pd.Series(valid_pred)
            output = pd.DataFrame(disaggregate_dict, dtype='float32')
            test_pred.append(output)

        return test_pred

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        sequence_length = self.sequence_length
        
        if method == 'train':
            processed_mains = []
            for mains in mains_lst:
                mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(), mains.values.std(), overlapping=True)
                processed_mains.append(pd.DataFrame(mains))

            processed_appliances = []
            for (app_name, app_df_lst) in submeters_lst:
                app_mean = self.appliance_params[app_name]['mean']
                app_std = self.appliance_params[app_name]['std']
                processed_app_dfs = []
                for app_df in app_df_lst:
                    appliance = self.normalize_data(app_df.values, sequence_length, app_mean, app_std, overlapping=True)
                    processed_app_dfs.append(pd.DataFrame(appliance))
                processed_appliances.append((app_name, processed_app_dfs))

            return processed_mains, processed_appliances

        if method == 'test':
            processed_mains = []
            for mains in mains_lst:
                mains = self.normalize_data(mains.values, sequence_length, mains.values.mean(), mains.values.std(), overlapping=True)
                processed_mains.append(pd.DataFrame(mains))
                
            return processed_mains

    def normalize_data(self, data, sequence_length, mean, std, overlapping):
        n = sequence_length
        padding_entries = sequence_length - (data.size % sequence_length)
        padding = np.array([0] * padding_entries)
        arr = np.concatenate((data.flatten(), padding), axis=0)

        if overlapping:
            windowed_x = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_x = arr

        windowed_x = (windowed_x - mean) / std
        return windowed_x.reshape((-1, sequence_length))

    def denormalize_data(self, data, mean, std):
        return (data * std + mean)

    def set_appliance_params(self, train_appliances):
        for (app_name, app_df_lst) in train_appliances:
            l = np.array(pd.concat(app_df_lst, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name:{'mean':app_mean, 'std':app_std}})

def train(appliance_name, model, train_mains, appliance_mains, epochs, batch_size, train_patience, checkpoint_interval=None, pretrain=False):
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)

    # model summary missing 
    
    train_mains, valid_mains, train_appliance, valid_appliance = train_test_split(train_mains, appliance_mains, 
                                                                                test_size=0.2, random_state=random_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss(reduction='mean')
    patience = 0
    best_loss = None

    train_dataset = TensorDataset(torch.from_numpy(train_mains).float(), torch.from_numpy(train_appliance).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    valid_dataset = TensorDataset(torch.from_numpy(valid_mains).float(), torch.from_numpy(valid_appliance).float())
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # SummaryWriter missing             

    for epoch in range(epochs):
        if(patience == train_patience):
            print("Validation Loss did not improve after {} epochs, call earlystopping.".format(train_patience))
            break

        st = time.time()

        model.train()
        for i, (batch_mains, batch_appliance) in enumerate(train_loader):
            if USE_CUDA:
                batch_mains = batch_mains.cuda()
                batch_appliance = batch_appliance.cuda()

            batch_pred = model(batch_mains)
            loss = loss_fn(batch_pred, batch_appliance)

            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        ed = time.time()

        model.eval()
        with torch.no_grad():
            cnt, loss_sum = 0, 0.0
            for i, (batch_mains, batch_appliance) in enumerate(valid_loader):
                if USE_CUDA:
                    batch_mains = batch_mains.cuda()
                    batch_appliance = batch_appliance.cuda()

                batch_pred = model(batch_mains)
                loss = loss_fn(batch_pred, batch_appliance)
                loss_sum += loss
                cnt += 1
        
        avg_loss = loss_sum / cnt

        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            best_state_dict = model.state_dict()
            path_state_dict = "./"+appliance_name+"_seq2seq_best_state_dict.pt"
            torch.save(best_state_dict, path_state_dict)
        else:
            patience += 1

        print("Epoch {}, Validation Loss: {}, Time Consumption: {}s".format(epoch+1, avg_loss, ed-st))

        # SummaryWriter missing

        if (checkpoint_interval != None) and ((epoch+1) % checkpoint_interval == 0):
            checkpoint = {"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch}
            path_checkpoint = "./"+appliance_name+"_seq2seq_checkpoint_epoch{}.pt".format(epoch+1)
            torch.save(checkpoint, path_checkpoint)

def test(model, test_mains, batch_size):
    if USE_CUDA:
        model = model.cuda()
    st = time.time()
    model.eval()

    test_dataset = TensorDataset(torch.from_numpy(test_mains).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    with torch.no_grad():
        for i, batch_mains in enumerate(test_loader):
            batch_main = batch_mains[0]
            if USE_CUDA:
                batch_main = batch_main.cuda()
            batch_pred = model(batch_main)
            if i == 0:
                output = batch_pred
            else:
                output = torch.cat((output, batch_pred), dim=0)
    
    ed = time.time()
    print("Inference Time Consumption: {}s.".format(ed-st))

    return output.to('cpu').numpy()

if __name__ =='__main__':
    experiment = {
    # Specify power type, sample rate and disaggregated appliance
    'power': {
        'mains': ['active'],
        'appliance': ['active']
    },
    'sample_rate': 6,
    'appliances': ['kettle'],
    # Universally no pre-training
    'pre_trained':False,
    # Specify algorithm hyper-parameters
    'methods':{"Seq2Seq":Seq2Seq({'n_epochs':5,'batch_size':256})},
    # Specify train and test data
    'train': {
        'datasets':{
        'ukdale': {
            'path': 'data/ukdale.h5',
            'buildings': {
                1: {
                        'start_time': '2013-06-01',
                        'end_time': '2013-07-01'
                }
            }
            },      
        }
    },
    'test': {
        'datasets':{
        'ukdale': {
            'path': 'data/ukdale.h5',
            'buildings': {
                1: {
                        'start_time': '2013-07-01',
                        'end_time': '2013-07-07'
                }
            }
            },
            
    },
    # Specify evaluation metrics
    'metrics': ['mae']
    }
    }

    api_experiment_1 = API(experiment)