from data.data_loader import Dataset_spectra_sim
from exp.exp_basic import Exp_Basic
from models.model import Transformer  

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

from utils.natural_sort_key import natural_sort_key

import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax,pad
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import matplotlib.pyplot as plt
import random


import warnings
# warnings.filterwarnings('ignore') #-----------------commented out to see warnings----->
import logging
import h5py
logger = logging.getLogger(__name__)

class Exp_transformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_transformer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'transformer':Transformer

        }
        if self.args.model=='transformer' or self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model in ['transformer','informer'] else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'mixture_ftir_data':Dataset_spectra_sim,
            'spectra_sim':Dataset_spectra_sim,

        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            # Data = Dataset_Pred
        else:
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq  #-------Changed shuffle_flag to False from True---->
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            share_vocab = args.share_vocab 
        )
        logger.info(f"{flag}, {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        
        #-------KL Divergence Loss----------------->
        # criterion = nn.KLDivLoss(reduction="batchmean")

        return criterion

#==========Training of the model with train, validation and test====================
#-----------Train data for model update, Validation data for stopping of the model training, and
# ----------Test data for model accuracy check-------------------------------------   
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        self.args.output_attention = False
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        best_warmup_loss = float('inf')
        best_warmup_model_path = None
        # best_learning_rate = self.args.learning_rate

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mae, train_mse, train_rmse, train_mape, train_mspe=[],[],[],[],[]
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                # if iter_count>=2:
                #     break

                #-------Forward pass and loss calculation-------->
                model_optim.zero_grad()
                
                logger.debug(f"Batch Input --> batch_x: {batch_x.size()},{batch_x}\nbatch_y: {batch_y.size()},{batch_y}\n"
                             f"batch_x_mark: {np.shape(batch_x_mark)},{batch_x_mark}\nbatch_y_mark: {np.shape(batch_y_mark)},{batch_y_mark}")
                
                pred_padded, true,_ = self._process_one_batch(i,
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, is_training=True)
                
                loss = criterion(pred_padded, true)
                train_loss.append(loss.item())

                mae, mse, rmse, mape, mspe = metric(pred_padded.detach().cpu().numpy(), true.detach().cpu().numpy())
                train_mae.append(mae); train_mse.append(mse); train_rmse.append(rmse); train_mape.append(mape); train_mspe.append(mspe)
                
                
                if (i+1) % 100==0:
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)

                    logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}\n".format(i + 1, epoch + 1, loss.item()) +
                            "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    
                    
                    iter_count = 0
                    time_now = time.time()
                
                #-------Backward pass and optimization-------->
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()


            logger.info("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            # print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            logger.info(f"Additional Metrics: [MAE: {np.average(train_mae):.2f}, MSE: {np.average(train_mse):.2f}, RMSE: {np.average(train_rmse):.2f}, MAPE: {np.average(train_mape):.2f}, MSPE: {np.average(train_mspe):.2f}]")
            
            
            if self.args.lradj=='type3':
                if epoch+1>self.args.warmup_epoch:
                    early_stopping(vali_loss, self.model, path)
                
                elif epoch+1==self.args.warmup_epoch:
                    if best_warmup_model_path:
                        self.model.load_state_dict(torch.load(best_warmup_model_path))
                        # self.args.learning_rate = best_learning_rate
                        model_optim = self._select_optimizer() # Reinitialize optimizer
                        logger.info(f"Loaded best model from: {best_warmup_model_path} ")


                else:
                    logger.info(f"Epoch:{epoch+1} Still in warmup phase, skipping early stopping critieria check")
                    
                    if vali_loss < best_warmup_loss:
                        best_warmup_loss = vali_loss
                        # best_learning_rate = model_optim.param_groups[0]['lr'] 
                        logger.info(f"Best warmup model loss: {best_warmup_loss} at epoch {epoch+1}")
                        best_warmup_model_path = os.path.join(path, 'best_warmup_model.pth')
                        torch.save(self.model.state_dict(), best_warmup_model_path)
                        logger.info(f"Best warm- up model updated and saved at: {best_warmup_model_path}")
            else:
                early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)


            if epoch % 5 == 0:
                plt.figure(figsize=(10, 6))
                if self.args.batch_size> 2:
                    rand_batch = random.sample(range(1, self.args.batch_size), 2)
                else:
                    rand_batch = 0
                plt.scatter(range(true[rand_batch,:,:].numel()), true[rand_batch,:,:].detach().cpu().numpy().flatten(),
                             label="True Values", alpha=0.7, marker='o')
                plt.scatter(range(pred_padded[rand_batch,:,:].numel()), pred_padded[rand_batch,:,:].detach().cpu().numpy().flatten(),
                             label="Predicted Values", alpha=0.7, marker='x')
                plt.xlabel("Wavenumber")
                plt.ylabel("Intensity")
                plt.title(f"True vs Predicted Values epoch {epoch + 1}")
                plt.legend()
                plt.grid(True)

                # Save the plot without displaying it
                plot_path = os.path.join(path, f"true_vs_pred_plot_epoch_{epoch + 1}.png")
                plt.savefig(plot_path)
                plt.close()

                logger.info(f"True vs Predicted plot for epoch {epoch + 1} saved at: {plot_path}")
            
        # Choose between best of warmup and after warmup  
        if np.abs(early_stopping.best_score)> best_warmup_loss:
            best_model_path = path+'/'+'best_warmup_model.pth'
            logger.info(f"========Best model is from warmup phase========")
        else:
            best_model_path = path+'/'+'checkpoint.pth'
            logger.info(f"========Best model is from AFTER warmup phase========")
        self.model.load_state_dict(torch.load(best_model_path))

        torch.save(self.model.state_dict(), path+'/'+'checkpoint.pth')

        return self.model
    
    
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            # if i>=2:
            #     break
            
            pred_padded, true,_ = self._process_one_batch(i,
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, is_training=True)
            
            loss = criterion(pred_padded.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
#=============Running on test data to get test statistics=======================

    def test(self, flag, setting, load):
        if flag == 'test':
            test_data, test_loader = self._get_data(flag='test')
        elif flag == 'train':
            test_data, test_loader = self._get_data(flag = 'train')
        # test_data, test_loader = self._get_data(flag='test')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
           
            total_params = sum(p.numel() for p in self.model.parameters())  # Total parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) # Trainable parameters

            logger.info(f"Total parameters: {total_params} & Trainable parameters: {trainable_params}")
            
        
        self.model.eval()
        
        preds = []
        trues = []
        all_attns_eigval = [] 
        self.args.output_attention = True
        
        start_time = time.time()
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            # if i>=0:
            #     break
         
            pred_padded, true, attns = self._process_one_batch(i,
                                        test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, is_training=True)
            
            preds.append(pred_padded.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            batch_eigenvalues = eigval_attn(attns)


            all_attns_eigval.append(batch_eigenvalues)
            
            # Save eigenvalue of attention tensors in chunks to release memory
            if (i + 1) % 5 == 0:
                chunk_folder_path = './results/' + setting +'/'+ 'attns_eigval_chunks/'
                if not os.path.exists(chunk_folder_path):
                    os.makedirs(chunk_folder_path)
                chunk_file_path = chunk_folder_path + f'attn_eigval_chunk_{i // 5 + 1}.h5'
                with h5py.File(chunk_file_path, 'w') as hf:
                    for ln in range(len(all_attns_eigval)):
                        hf.create_dataset('attns_eigval'+str(ln), data=all_attns_eigval[ln], compression='lzf')
                all_attns_eigval = []

                logger.info(f"Test Batch {i + 1}/{len(test_loader)} completed in time: {((time.time()-start_time)/60):.2f} min.")
            
            elif i == len(test_loader) - 1:
                chunk_folder_path = './results/' + setting +'/'+ 'attns_eigval_chunks/'
                if not os.path.exists(chunk_folder_path):
                    os.makedirs(chunk_folder_path)
                chunk_file_path = chunk_folder_path + f'attn_eigval_chunk_{i // 5 + 2}.h5'
                with h5py.File(chunk_file_path, 'w') as hf:
                    for ln in range(len(all_attns_eigval)):
                        hf.create_dataset('attns_eigval'+str(ln), data=all_attns_eigval[ln], compression='lzf')
                        # hf.create_dataset('attns_eigval', data=np.array(batch_eigenvalues), compression='lzf')
                all_attns_eigval = []

                logger.info(f"Test Batch {i + 1}/{len(test_loader)} completed in time: {((time.time()-start_time)/60):.2f} min.")
        
        preds = np.array(preds)
        trues = np.array(trues)
        
        # Read eigenvalue of attention chunks back
        chunk_folder_path = './results/' + setting + '/' + 'attns_eigval_chunks/'
        all_attns_eigval = []
        
        chunk_files = sorted([f for f in os.listdir(chunk_folder_path) if f.startswith('attn_eigval_chunk_') and f.endswith('.h5')], key=natural_sort_key)
        
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_folder_path, chunk_file)
            # chunk_eigval = []
            with h5py.File(chunk_path, 'r') as hf:
                chunk_attns = []
                for key in hf.keys():
                    chunk_attns.append(hf[key][:])
                chunk_attns = np.concatenate(chunk_attns, axis=1)
                all_attns_eigval.append(chunk_attns)
        
        eigval_layer=[]
        layer = len(all_attns_eigval[-1])
        
        for i in range(layer):
            attns_l = [item[i] for item in all_attns_eigval]
            attns_concat = np.concatenate(attns_l, axis=0)
            eigval_layer.append(attns_concat)

        logger.info(f'Total eigenvalues shape for each layer: {eigval_layer[-1].shape}')
        # logger.info(f'attention shape: {all_attns.shape}')
        logger.info(f'test shape:{preds.shape},{trues.shape}')
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        logger.info(f'test shape:{preds.shape},{trues.shape}')
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logger.info(f'mse:{mse:.2f}, mae:{mae:.2f}, rmse:{rmse:.2f}, mape:{mape:.2f}, mspe:{mspe:.2f}')
        # print('mse:{}, mae:{},rmse:{}, mape:{}, mspe:{}'.format(mse, mae,rmse, mape, mspe))

        np.save(folder_path+'metrics_testdata.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred_testdata.npy', preds)
        np.save(folder_path+'true_testdata.npy', trues)
        # np.save(folder_path + 'attns_testdata.npy', all_attns)  # Save attention tensors
        
        for i, eigvals in enumerate(eigval_layer):
            np.save(folder_path + f'eigenvalues_layer_{i+1}_testdata.npy', eigvals)   # Save eigenvalues for each layer


        # Delete the chunk files to save space
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_folder_path, chunk_file)
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
                logger.info(f"Deleted chunk file: {chunk_path}")

        return


    def _process_one_batch(self, batch,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, is_training=True):   
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
    

        attns = []

        if is_training:
            
        
            dec_inp = batch_y.float().to(self.device) 
            
            
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        
                        out,attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                if self.args.output_attention:
                    out,attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            
            outputs = out[0] #dec_out[:,:-1,:] for decoder included architecture--------------->

            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)

            # ----------log softmax of the predicted logits----------------->
            y_pred = outputs#log_softmax(outputs, dim=-1)

        else:
            raise NotImplementedError  
        
            
               

        f_dim = -1 if self.args.features=='MS' else 0
        
        y_true = out[1][:, :, f_dim:].to(self.device)
        #-------------------------------------------------------------------------------------------------->
        return y_pred, y_true,attns 

def eigval_attn(attns):
    eigvals_list = []
    for attn in attns:
        batch_eigvals = []
        for b in range(attn.size(0)):  # Loop over batch size
            head_eigvals = []
            for h in range(attn.size(1)):  # Loop over number of heads
                try:

                    diag_abs = np.diag(attn[b, h].detach().cpu().numpy())**2
                    scaled_rownorm = np.sqrt(diag_abs) #np.log(row_norm+1)-np.log(diag_abs+1) #row_norm / (diag_abs)
                    head_eigvals.append(scaled_rownorm)

                except:
                    epsilon = 1e-5
                    head_eigvals.append(np.array(epsilon * np.random.rand(attn.size(-1))))
                    logger.warning(f"Eigenvalues couldn't be calculated for batch {b+1}, head {h+1} due to ill-conditioning, using small random values instead.")
            batch_eigvals.append(head_eigvals)
        eigvals_list.append(batch_eigvals)
    return eigvals_list

def label_smoothing(targets, vocab_size, smoothing=0.1):
    """
    Apply label smoothing to the targets.
    
    Args:
        targets (torch.Tensor): The target tensor of shape (batch_size, seq_len).
        vocab_size (int): The size of the vocabulary.
        smoothing (float): The smoothing value.
    
    Returns:
        torch.Tensor: The smoothed target tensor of shape (batch_size, seq_len, vocab_size).
    """
    confidence = 1.0 - smoothing
    low_confidence = smoothing / (vocab_size - 1)
    
    # Create a tensor of shape (batch_size, seq_len, vocab_size) filled with low_confidence
    smoothed_targets = torch.full(size=(targets.size(0), targets.size(1), vocab_size), fill_value=low_confidence).to(targets.device)
    
    # Fill the positions corresponding to the target indices with confidence
    smoothed_targets.scatter_(2, targets, confidence)
    
    return smoothed_targets
