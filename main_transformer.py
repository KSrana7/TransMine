##

import numpy as np
import argparse
import os
import torch
from exp.exp_transformer import Exp_transformer 
# from utils.build_vocabulary import build_vocab
from utils.prepare_data import prepare_data
from utils.test_analysis import test_analysis
from utils.Kinetics import Kinetics

# Configure logger

import logging
import ast


mode= [
    'train',
    'test',
    'kinetics'
       ] 



file_log = 'TransformerLog.log'

# Clear the log file at the start of each run
with open(file_log, "w"):
    pass

logging.basicConfig(level=logging.INFO,
                    format="[(%(asctime)s)-%(name)s(%(filename)s)-%(levelname)s]-Line:%(lineno)d \n %(message)s",#'\n >> (%(asctime)s) - %(name)s (%(filename)s) - %(levelname)s - Line:%(lineno)d \n %(message)s',
                    handlers=[
                        logging.FileHandler(file_log),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


logger.info(f'====================================Mode:{mode}==================================')

parser = argparse.ArgumentParser(description='[Transformer] for spectral data')

args = parser.parse_args()

args.deltaT = [50]  #----Shift length----->
args.h_n = [8] #----Number of heads----->   
args.e_n = [2] #----Number of encoder layers----->
args.d_model=[512] #----Dimension of model/embedding----->
args.des = ["system2"]#"Series","Series_customAppend2","Series_CustomInput"]


for idx, (ii, hn, en, dm, des) in enumerate(zip(args.deltaT, args.h_n, args.e_n, args.d_model, args.des)):
    print(f"Iteration {idx}: deltaT:{ii}, hn:{hn}, en:{en}, dm:{dm}, des:{des}")
    
    print(f"deltaT:{ii},hn:{hn},en:{en},dm:{dm}")
    #args = parser.parse_args()
    args = argparse.Namespace(task='spectral',model='transformer',data='mixture_ftir_data',root_path='./data/Rxn_FTIR2/',
        data_name='MixtureSpectra_processed_0_13200_11674_400.csv',features='M',target='',freq='',checkpoints='./checkpoints/',seq_len=1,label_len=0,
        pred_len=1,enc_in='',c_out='',d_model=512,n_heads=8,e_layers=2,d_ff=2048, 
        factor=5,padding=0,distil=False,dropout=0.05,attn='full',embed='',activation='gelu',output_attention=True,
        do_predict=False,mix=True,cols=None,num_workers=0,itr=2,train_epochs=50,batch_size=32,patience=5,learning_rate=0.001,
        des='system2',loss='mse',lradj='type3',use_amp=False,inverse=False,use_gpu=True,gpu=0,use_multi_gpu=False,devices='0,1,2,3',share_vocab=True                                                                         
    )                                                                                                          

    k = ii
    args.deltaT= k
    args.n_heads = hn
    args.e_layers = en
    args.d_model = dm
    args.d_ff = 4*dm
    args.des = des

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    # Data Information

    data_parser_spectral = {
    'mixture_ftir_data':{'data':args.data_name, #'mixture_ftir_data.csv',
                    'Source':['src-train.txt','src-val.txt','src-test.txt','src-pred.txt'],
                    'Target':['tgt-train.txt','tgt-val.txt','tgt-test.txt','tgt-pred.txt']} ,
    'spectra_sim':{'Source':['src-train.txt','src-val.txt','src-test.txt','src-pred.txt'],
                    'Target':['tgt-train.txt','tgt-val.txt','tgt-test.txt',''],  #,'tgt-pred.txt'
                    }, #----Spectral data details: data sequence is train,val,test ---->             
    }


    logger.info(f'==============================================================')

    
    if args.task == 'spectral' and args.data in data_parser_spectral.keys():
        data_info = data_parser_spectral[args.data]
        args.data_path = data_info['Source']
        args.target = data_info['Target']
        
        # Prepare data for translation task
        file_path = args.root_path+data_info['data']


        # Create data sets
        train_data, val_data, test_data, train_target, val_target, test_target,scale = prepare_data(file_path, k)
        
        src_len = np.shape(train_data)[1]
        tgt_len = np.shape(train_target)[1]

        # Save the data to txt files
        np.savetxt(args.root_path+data_info['Source'][0], train_data)
        np.savetxt(args.root_path+data_info['Target'][0], train_target)
        np.savetxt(args.root_path+data_info['Source'][2], test_data)
        np.savetxt(args.root_path+data_info['Target'][2], test_target)
        np.savetxt(args.root_path+data_info['Source'][1], val_data)
        np.savetxt(args.root_path+data_info['Target'][1], val_target)


        if args.do_predict:
            np.savetxt(args.root_path+data_info['Source'][3], test_data, fmt='%d')
            np.savetxt(args.root_path+data_info['Target'][3], test_target, fmt='%d')
        
        args.enc_in, args.dec_in, args.c_out = 1,1,1 
        
        
        if args.lradj=='type3':
            args.warmup_epoch = int(0.3*args.train_epochs) #---30% of epochs for warmup(exploration)----->
            args.warmup_steps = max(1,args.warmup_epoch*len(train_data)//args.batch_size) 
            args.steps = 1*len(train_data)//args.batch_size
            args.learning_rate = args.d_model**(-0.5)*min(args.steps**(-0.5),args.steps*args.warmup_steps**(-1.5))
            logger.info(f'Intial learning rate for Noah scheduler: {args.learning_rate} with warmup-steps: {args.warmup_steps}')


    else:        logger.error(f"Data: {args.data} not recognized. Please check the data_parser_spectral dictionary for available datasets.")


    args.detail_freq = args.freq
    args.freq = args.freq[-1:]


    logger.debug(f"Args in experiment: \n {args}")

    Exp = Exp_transformer


    # setting record of experiments
    setting = '{}_{}_F{}_SL{}_LL{}_PL{}_Dmodel{}_head{}_Elayer{}_Dff{}_DES{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_ff,  
                args.des, ii)

    exp = Exp(args) # set experiments
    
    if 'train' in mode:
        #============================Train the Model===================================
        logger.info('=================Start Training :{}================'.format(setting))
        exp.train(setting)
    else:
        logger.info(f'====================Skipping Training as mode:{mode}=================')
    
    if 'test' in mode:    

        #=============Running on test data to get test statistics=======================
        logger.info('=================Testing :{}======================='.format(setting))

        # try:
        flag = 'train' #---test on train data
        load = True
        exp.test(flag, setting, load)
        # except:
        #     logger.warning(f"TESTING ERROR ON {setting}, SKIPPING IT")
        #     pass

        #=============Analysing test results=======================
        logger.info('=================Analysing :{}======================='.format(setting))

        Result_path ='./results'
        layer_num = args.e_layers

        test_analysis(Result_path,  layer_num, scale, setting)
    else:
        logger.info(f'====================Skipping Testing as mode:{mode}=================')


    if 'kinetics' in mode:    

        #=============Running on data to get kinetics=======================
        logger.info('=================Kinetics Estimation :{}======================='.format(setting))

        Data = (train_data,train_target)
        Result_path ='./results'
        layer_num = args.e_layers
        view = input("Enter view as [('layer#','head#')] (e.g., [('2','1')] or 'overall'): ")
        try:
            # Try to evaluate as Python literal (list/tuple)
            view = ast.literal_eval(view)
        except Exception:
            # If not a valid literal, keep as string
            pass

        activity_thresh = float(input("Enter activity threshold (e.g., 0.005): ") or 0.005)
        CNN_compression_ratio=2  #----Ratio by which CNN layer compresses Input to Embeddding dimesion across wavenumber-----> 

        args.mode='manual' # 'automatic' or 'manual'

        if args.mode == 'automatic':
            args.n_components=(1,7)
        elif args.mode == 'manual':
            zone_range_hr = input("Enter time range of each zone as list of tuple: ") # [(-20.0,4.5),(4.5,63.3),(29.0,220.0),(97.6,220.0)]
            try:
                # Try to evaluate as Python literal (list/tuple)
                zone_range_hr = ast.literal_eval(zone_range_hr)
                args.zone_range_hr = zone_range_hr
            except Exception as e:
                print(f"Error: Invalid input for zone_range_hr: {zone_range_hr} with error:{e}")
                raise
            
            # zone_var_med =  [[775.0,3055.0],[2925.0],[695.0,765.0],[895.0,985.0]]  # Characteristic bands of each zone
            
            # zone_var_med_all = [[775.0, 3055.0], [695.0, 2925.0], [695.0, 765.0, 3055.0], [695.0, 765.0, 895.0, 985.0,1550.0, 3055.0]]  # Characteristic + common bands of each zone

        Kinetics(Data,args, Result_path,  layer_num, scale,view, activity_thresh,CNN_compression_ratio, setting)


    else:
        logger.info(f'====================Skipping Kinetics Estimation as mode:{mode}=================')

    if args.do_predict:
        logger.info('==============Predicting :{}===================='.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

# # Close the logger
# for handler in logger.handlers:
#     handler.flush()
#     handler.close()
#     logger.removeHandler(handler)

