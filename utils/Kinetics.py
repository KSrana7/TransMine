import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tck.TCK import TCK
from sklearn.linear_model import LinearRegression
from exp.exp_transformer import Exp_transformer
import torch
from scipy.signal import find_peaks


from sklearn.mixture import GaussianMixture
import argparse
from collections import defaultdict
import json
import logging
import ast


logger = logging.getLogger(__name__)

class Kinetics:
    def __init__(self, Data,args, result_path, layers, scale, view,activity_thresh,CNN_compression_ratio, setting):
        """
        Args:
            result_path(string): Path to the result folder.
            layer_num(int): Layer number to analyze.
        """
        self.Data = Data
        self.args = args
        self.result_path = result_path
        self.layers = layers
        self.scale = scale
        self.view = view

        self.activity_thresh = activity_thresh
        self.CNN_compression_ratio = CNN_compression_ratio
        self.setting = setting

        '''---------------------------Parameters-------------------->'''
        self.plot_dir = os.path.join(result_path, setting, 'Kinetics',args.mode)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.wavenumber =scale[0]
        self.time = scale[1]

        self.flag = 'train'
        self.prob_thresh=0.9   #    prob_thresh (float): Posterior probability threshold (default 0.9). &   activity_thresh (float): Threshold to consider variable as active (default 1e-5).
        self.save= True
        self.plot=True
        self.cover_sigma= 1.645 # factor to multiple for STD to cover the area of each Gaussian (for 90% confidence interval)
        self.cover_area= 1.000   # area to cover for each Gaussian 

        self.delta =1  #-------delta T for dx/dt

        self.kinetics()

    '''---------------------------Supporting Functions-------------------->'''
    def extract_temporal_gaussians(self,data, plot=True):
            """
            Fit a Gaussian Mixture Model to the time-series data and extract dominant components.

            Parameters:
                data: 2D array of shape (L, T) where each column is a sample (L variables at one time).
                n_components: int, number of Gaussian components to fit (default 3).
                plot: bool, if True will plot the Gaussian component curves over time.

            Returns:
                List of dicts, each with keys 'mean_time', 'std_time', 'weight' for each component,
                sorted by descending weight.
            """
            # Prepare data: each row is a timepoint sample of length L
            X = data.T  # shape (T, L)
            T, L = X.shape
            layer,head = self.view[0]
            n_components = self.args.n_components
            time_label = np.round(self.time[self.args.batch_size:-self.args.batch_size]/60, 2)
            spectral_label = self.wavenumber[:-1:self.CNN_compression_ratio].tolist()
            # Fit GMM
            # aic_values = []
            # bic_values = []
            log_likelihoods = []

            for n in range(n_components[0], n_components[1] + 1):
                gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
                gmm.fit(X)
                log_likelihoods.append(gmm.score(X))#log_likelihoods.append(gmm.score(X) * len(X))  # total log-likelihood


            # Compute cumulative change in log-likelihood
            log_likelihoods = np.array(log_likelihoods)
            # cumulative_change = (log_likelihoods)/log_likelihoods[0]

            # Calculate "score explained" (relative improvement) in percentage
            score_explained = 100 * (log_likelihoods) / (log_likelihoods[-1] + 1e-10)

            # Plot
            # High-quality figure for scientific publication: GMM Log-Likelihood Score vs Number of Components
            plt.figure(figsize=(12, 8), dpi=300)
            x_vals = list(range(n_components[0], n_components[1] + 1))
            plt.plot(x_vals, log_likelihoods, label='Log-Likelihood Score', color='#2c7bb6', marker='o', markersize=8, linewidth=2.5)
            plt.xlabel('Number of GMM Components', fontsize=18, weight='bold')
            plt.ylabel('GMM Log-Likelihood Score', fontsize=18, weight='bold')
            plt.title('GMM Log-Likelihood Score vs Number of Components', fontsize=20, weight='bold')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=15, frameon=False)
            plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
            sns.despine()
            plt.tight_layout()
            if self.save:
                plt.savefig(os.path.join(self.plot_dir, f'GMM_scores_{layer}_{head}.png'), bbox_inches='tight')
            # plt.show()

            # Plot score explained (percentage)
            # High-quality figure for scientific publication: Percentage of Highest Score Explained by Number of GMM Components
            plt.figure(figsize=(12, 8), dpi=300)
            x_vals = list(range(n_components[0], n_components[1] + 1))
            plt.plot(x_vals, score_explained, marker='o', color='#d7191c', linewidth=2.5, markersize=8, label='Score Explained (%)')
            plt.xlabel('Number of GMM Components', fontsize=18, weight='bold')
            plt.ylabel('Score Explained (%)', fontsize=18, weight='bold')
            plt.title('Percentage of Highest Score Explained by Number of GMM Components', fontsize=20, weight='bold')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=15, frameon=False)
            plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
            sns.despine()
            plt.tight_layout()
            if self.save:
                plt.savefig(os.path.join(self.plot_dir, f'GMM_score_explained_{layer}_{head}.png'), bbox_inches='tight')
            # plt.show()
        

            n_components = int(input("Select the number of GMM components: "))

            best_gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
            best_gmm.fit(X)

            # Best GMM selected based on lowest BIC
            logger.info(f"Optimal number of components selected: {best_gmm.n_components}")

            # Get posterior probabilities for each time point (T x K)
            # n_components = best_gmm.n_components
            posterior_probs = best_gmm.predict_proba(X)

            # Get component means (K x L): each mean vector shows time series activity pattern

            means = best_gmm.means_

            # Plot time-series co-occurrence for each component
            # High-quality figure for scientific publication: Posterior probabilities over time
            fig, axs = plt.subplots(n_components, 1, figsize=(14, 2.5 * n_components), sharex=True, 
                dpi=300, constrained_layout=True)
            time = np.arange(X.shape[0])

            for k in range(n_components):
                axs[k].plot(time, posterior_probs[:, k], color=f'C{k}', lw=2.5, label=f'Component {k+1}')
                axs[k].fill_between(time, posterior_probs[:, k], color=f'C{k}', alpha=0.18)
                axs[k].set_ylabel(f'$P$(Comp {k+1})', fontsize=14)
                axs[k].set_xlim(time[0], time[-1])
                axs[k].set_ylim(0, 1)
                axs[k].legend(loc='upper right', fontsize=12, frameon=False)
                axs[k].tick_params(axis='both', which='major', labelsize=13)
                axs[k].grid(True, linestyle='--', linewidth=0.7, alpha=0.7, which='both')
                if k < n_components - 1:
                    axs[k].set_xticklabels([])

            axs[-1].set_xlabel("Time", fontsize=15)
            # Set x-axis tick labels as time_label, but only 50 ticks
            num_ticks = 50
            tick_indices = np.linspace(0, len(time_label) - 1, num=num_ticks, dtype=int)
            axs[-1].set_xticks(tick_indices)
            axs[-1].set_xticklabels([time_label[i] for i in tick_indices], rotation=90, fontsize=12)
            plt.suptitle("Temporal Posterior Probability of Each GMM Component", fontsize=18, weight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            sns.despine()
            if self.save:
                plt.savefig(os.path.join(self.plot_dir, f'GMM_posterior_{layer}_{head}.png'), bbox_inches='tight')
            # plt.show()

            # Plot heatmap of component means to show time series co-activation
            # Arrange subplots in a square layout
            for k in range(n_components):
                plt.figure(figsize=(12, 8), dpi=300)  # High-quality figure for scientific publication
                plt.scatter(np.arange(L), means[k], color='#1f77b4', s=40, edgecolor='black', alpha=0.85)
                plt.title(f"GMM Component {k+1}: Mean Activity Across Wavenumbers", fontsize=18, weight='bold')
                plt.xlabel("Wavenumber (cm⁻¹)", fontsize=16)
                plt.ylabel("Mean Activity", fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.5, linewidth=1)
                # Set x-ticks to show wavenumber labels, but not too many
                tick_step = max(1, len(spectral_label) // 20)
                tick_indices = np.arange(0, len(spectral_label), tick_step)
                plt.xticks(tick_indices, [str(int(spectral_label[i])) for i in tick_indices], rotation=90, fontsize=12)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                sns.despine()
                if self.save:   
                    plt.savefig(os.path.join(self.plot_dir, f'GMM_components_{layer}_{head}_{k+1}.png'), bbox_inches='tight')
                # plt.show()

            # Responsibilities (posterior probabilities), shape (T, k)
            resp = best_gmm.predict_proba(X)
            weights = best_gmm.weights_
            T = data.shape[1]
            time_idx = np.arange(T)

            # Log-likelihood of the fitted GMM
            # log_likelihood = best_gmm.score(X) * X.shape[0]
            # print(f"GMM Log-Likelihood: {log_likelihood}")

            # Compute weighted mean and std in time for each component
            results = []
            for k in np.argsort(weights)[::-1]:  # sort components by weight (largest first)
                w = weights[k]
                r = resp[:, k]
                mean_time = np.sum(time_idx * r) / np.sum(r)
                var_time = np.sum(((time_idx - mean_time)**2) * r) / np.sum(r)
                std_time = np.sqrt(var_time)

                strong_vars, _ = find_peaks(means[k], prominence=1 * np.std(means[k]) * 2)

                results.append({'mean_time': mean_time, 'std_time': std_time, 'weight': w, 'strong_vars': strong_vars})

            # Optional: plot the Gaussian curves (density over time) for each component
            if plot:
                t = np.arange(T)
                plt.figure(figsize=(12, 8), dpi=300)  # High-quality figure for scientific publication
                for comp in results:
                    mt, st, wt = comp['mean_time'], comp['std_time'], comp['weight']
                    # Compute the (normalized) Gaussian PDF weighted by the mixture weight
                    density = wt * (np.exp(-0.5 * ((t - mt) / st)**2) / (st * np.sqrt(2 * np.pi)))
                    plt.plot(t, density, label=f"mean={mt:.1f}", linewidth=2)
                plt.xlabel('Time index', fontsize=16)
                plt.ylabel('Mixture density', fontsize=16)
                plt.title('Gaussian Mixture Components Over Time', fontsize=18, weight='bold')
                plt.legend(fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                num_ticks = 50
                tick_indices = np.linspace(0, len(time_label) - 1, num=num_ticks, dtype=int)
                plt.xticks(ticks=tick_indices, labels=[time_label[i] for i in tick_indices], rotation=90, fontsize=12)
                plt.xlabel('Time (min)', fontsize=16)
                sns.despine()
                plt.tight_layout()
                if self.save:
                    plt.savefig(os.path.join(self.plot_dir, f'GMM_{n_components}_{layer}_{head}.png'), bbox_inches='tight')

                # plt.show()

            return results
    
    def segment_linear_sequences(self, lst, delta, tol=1e-3):
        groups = []
        current_group = []

        for i, val in enumerate(lst):
            if not current_group:
                current_group.append(val)
            else:
                expected = current_group[-1] + delta
                if abs(val - expected) < tol:
                    current_group.append(val)
                else:
                    groups.append((current_group, np.median(current_group)))
                    current_group = [val]

        if current_group:
            groups.append((current_group, np.median(current_group)))

        return groups

    #---------------------------------------------------------------------->
    #%%

    def kinetics(self):
        """
        Args:
            result_path(string): Path to the result folder.
            layer_num(int): Layer number to analyze.
        """
        Data = self.Data 
        result_path = self.result_path  
        layers = self.layers 
        view = self.view 
        # n_components = self.n_components 
        CNN_compression_ratio = self.CNN_compression_ratio 
        setting = self.setting 
        plot_dir = self.plot_dir
        wavenumber = self.wavenumber 
        time = self.time 
        plot=self.plot
        cover_sigma = self.cover_sigma # factor to multiple for STD to cover the area of each Gaussian (for 90% confidence interval)
        cover_area = self.cover_area  # area to cover for each Gaussian 


        if view=='overall':
            self.view = [('All','All')]
            filename = os.path.join(result_path, setting, f'eigenvalues_layer_{1}_testdata.npy')
            eigenvalues1 = np.load(os.path.abspath(filename))
            cleaned_eigenvalues = np.nan_to_num(eigenvalues1, nan=1e17, posinf=1e17, neginf=-1e17)
            overall_data = cleaned_eigenvalues.sum(axis=1)  # Sum across H
            # overall_data = (overall_data - overall_data.min(axis=1).min()) / (overall_data.max(axis=1).max() - overall_data.min(axis=1).min())

            for layer_num in range(1,layers):
                filename = os.path.join(result_path, setting, f'eigenvalues_layer_{layer_num+1}_testdata.npy')
                eigenvalues1 = np.load(os.path.abspath(filename))
                cleaned_eigenvalues = np.nan_to_num(eigenvalues1, nan=1e17, posinf=1e17, neginf=-1e17)
                concatenated_eigenvalues = cleaned_eigenvalues.sum(axis=1)  # Sum across H
                # concatenated_eigenvalues = (concatenated_eigenvalues - concatenated_eigenvalues.min(axis=1).min()) / (concatenated_eigenvalues.max(axis=1).max() - concatenated_eigenvalues.min(axis=1).min())
                overall_data = overall_data + concatenated_eigenvalues  # Sum across layers
        
        elif isinstance(view, list) :
            for layer,head in view:
                layer_num = int(layer)
                head_num = int(head)
                filename = os.path.join(result_path, setting, f'eigenvalues_layer_{layer_num}_testdata.npy')
                eigenvalues1 = np.load(os.path.abspath(filename))
                cleaned_eigenvalues = np.nan_to_num(eigenvalues1, nan=1e17, posinf=1e17, neginf=-1e17)
                concatenated_eigenvalues = cleaned_eigenvalues[:,head_num-1,:]  # get head H

                #-----------------To invert the transmittance activation to absorbance----------------->
                # concatenated_eigenvalues= np.exp(-concatenated_eigenvalues)  # Exponentiate to get original values
                # self.activity_thresh = np.exp(-self.activity_thresh)
                overall_data = concatenated_eigenvalues
        else:
            raise ValueError("Invalid view option. Use 'overall' or 'layer'.")
        
        
        #'''Time-series Cluster Kernel (TCK)'''

        '''------------------------------------Gaussian Mixture models across time dimension----------------'''



        data = overall_data.transpose(1,0)  

        #=========================Fit GMM and extract components====================================>
        if self.args.mode == 'automatic':
            
            components = self.extract_temporal_gaussians(data, plot=True)  #Time series kernel clustering
            components = sorted(components,key=lambda x:x['mean_time'])
            print(components)

            strong_wavenumber = []
            strong_wavenumber_ind = []
            for comp in components:
                strong_wavenumber.append([wavenumber[CNN_compression_ratio*x] for x in comp['strong_vars']])
                strong_wavenumber_ind.append([x for x in comp['strong_vars']])
                print(f"GMM Component mean {np.round(comp['mean_time'])}: Wavenumbers Matched: {[wavenumber[2*x] for x in comp['strong_vars']]}")

            '''----------------------------------------------------------------'''
            

            '''=============Linear regression to idnetify the kinetic parameters==============='''
            zone_range = [(max(0, int(comp['mean_time'] - cover_sigma * comp['std_time'])), 
                    min(overall_data.shape[0], int(comp['mean_time'] + cover_sigma * comp['std_time']))) 
                    for comp in components]
            
            zone_var =strong_wavenumber.copy()
            zone_var_ind =strong_wavenumber_ind.copy()

            #----------------Find different spectral region(for some bands) and for each of these regions find their medians----------------->

            zone_var_med = []
            zone_var_med_ind = []
            delta_l = CNN_compression_ratio*np.diff(wavenumber)[0]  

            for ii in range(len(zone_var)):
                out = self.segment_linear_sequences(zone_var[ii], delta_l,0.1)
                out_ind = self.segment_linear_sequences(zone_var_ind[ii], 1.0,0.1)
                med_list = []
                med_list_ind = []
                for (group, median),(_,median2) in zip(out,out_ind):
                    print(f"Group: {ii}, Median_wavenumber: {median}, Median_index: {median2}")
                    med_list.append(median)
                    med_list_ind.append(median2)
                zone_var_med.append(med_list)
                zone_var_med_ind.append(med_list_ind)
            #-------------------------------------------------------------------------------------> 
            logger.info(f"Zone_var_med:{zone_var_med} &\n Zone_var_ind:{zone_var_ind}")

            row_ind = sorted(list(set(elem for sublist in zone_var_med_ind for elem in sublist)))
            row_label = [wavenumber[int(CNN_compression_ratio*i)] for i in row_ind]
        
            row_ind=[int(i) for i in row_ind]
        
        elif self.args.mode == 'manual':
            #-------------------------Manual spectral band picking------------------------------>
            zone_range_hr = self.args.zone_range_hr #[(-1.7,0.0),(0.0,4.0),(4.5,18.3),(8.9,18.3)]
            time_range=(0,self.time.shape[-1]-self.args.deltaT)
            if self.flag == 'train' or self.flag == 'test':
                time_range=(0,time_range[1]-time_range[1]%self.args.batch_size)  # Time range in minutes
            trimmed_time = self.time[time_range[0]:time_range[1]]

            # For each element in zone_range_hr, multiply by 60 and find the index in trimmed_time closest to that value
            zone_range = []
            for start, end in zone_range_hr:
                start_val = start * 60
                end_val = end * 60
                start_idx = np.abs(trimmed_time - start_val).argmin()
                end_idx = np.abs(trimmed_time - end_val).argmin()
                zone_range.append((start_idx, end_idx))
            # For each element of zone_range, in that range identify the rows which are always above the activity_thresh
            zone_ind = []
            zone_var=[]
            for idx, (start, end) in enumerate(zone_range):
                zone_data = data[:, start:end]
                # Identify rows where most values(90%) are greater than activity_thresh
                active_rows = np.where(np.mean(zone_data > self.activity_thresh, axis=1) >= 0.8)[0]
                zone_ind.append(active_rows.tolist())
                zone_var.append([wavenumber[int(CNN_compression_ratio*i)] for i in active_rows])
            
            logger.info(f"Zone_var:{zone_var} &\n Zone_ind:{zone_ind}")


            zone_var_med = input(f"Characteristic bands of each zone as list of list [[],[]]: ")    #[[630,1440,3000,3040],[630,1720,2980,3250],[1030,1070,3250,3410],[950,1210]]  # Characteristic bands of each zone
            
            zone_var_med_all = input(f"All representative bands of each zone as list of list [[],[]]: ")  
                                # [[630.0, 1440.0, 2820.0, 2960.0, 3000.0, 3040.0],
                                # [1180.0, 1720.0, 2820.0, 2980.0],
                                # [1030.0, 1070, 1210, 2820.0, 2960, 3250.0, 3410.0],
                                # [950, 1030.0, 1070, 1210, 2820.0, 2960.0, 3220.0, 3450.0]]  # Characteristic + common bands of each zone

            try:
                zone_var_med = ast.literal_eval(zone_var_med)
                zone_var_med_all = ast.literal_eval(zone_var_med_all)
            except Exception as e:
                logger.error(f"Error parsing zone_var_med or zone_var_med_all: {e}. Please provide a valid list of lists (e.g., [[1,2],[3,4]]).")
                raise 
            
            zone_var_med_ind = [[int(wavenumber.tolist().index(elm)/2) for elm in item] for item in zone_var_med]   #[11.5, 239.0], [225.5], [3.5, 10.5], [3.5, 11.0, 23.5]
            
            zone_var_med_ind_all = [[int(wavenumber.tolist().index(elm)/2) for elm in item] for item in zone_var_med_all]

            logger.info(f"Zone_var_selected:{zone_var_med} ")
            #----------------------------------------------------------------------------------->
            row_ind = sorted(list(set(elem for sublist in zone_var_med_ind_all for elem in sublist)))
            row_label = [wavenumber[int(CNN_compression_ratio*i)] for i in row_ind]
        
            row_ind=[int(i) for i in row_ind]

        else:
            raise ValueError("Invalid mode. Use 'automatic' or 'manual'.")
        


        '''=============================Corelation of the spectral peaks with other peaks to identify the likely dependency of transformation====== '''
        # [[self.scale[1].tolist()[ii] for ii in item] for item in zone_range]


        Exp = Exp_transformer
        exp = Exp(self.args)
        flag = self.flag

        if flag == 'test':
            test_data, test_loader = exp._get_data(flag='test')
        elif flag == 'train':
            test_data, test_loader = exp._get_data(flag = 'train')
        # test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))
        
        total_params = sum(p.numel() for p in exp.model.parameters())  # Total parameters
        trainable_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad) # Trainable parameters
        # layer = exp.model.encoder.enc_layer[0].attention 
        # W_o = layer.out_projection.weight.data  #-----------Projection Layer weights--------------------->

        logger.info(f"Total parameters: {total_params} & Trainable parameters: {trainable_params}")
            
        
        exp.model.eval()
        
        preds = []
        trues = []
        self.args.output_attention = True
        focal_attention = []

    
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            # if i>=0:
            #     break
         
            pred_padded, true, attns = exp._process_one_batch(i,
                                        test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, is_training=True)
            
            preds.append(pred_padded.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            if self.view[0][0]=='All' and self.view[0][1]=='All':
                layer_attn = []
                for ik in range(1,len(attns)):
                    layer_attn.append(attns[ik].detach().cpu().numpy()[:,:,row_ind][:,:,:,row_ind].sum(axis=1))
                
                focal_attention.append(np.sum(layer_attn,axis=0)) # save only specified aattention of layer, head and spectral regions
            else:
                focal_attention.append(attns[int(layer)-1][:,int(head)-1,:].detach().cpu().numpy()[:, row_ind, :][:, :, row_ind])

            if i % 10 == 0:
                logger.info(f"Processing batch {i + 1}/{len(test_loader)}")
            
            
        focal_attention_append = np.concatenate(focal_attention, axis=0)
        zone_var_med


        for ii in range(len(zone_range)):

            zone_data = focal_attention_append[zone_range[ii][0]:zone_range[ii][1],:,:]
            mean_corr = zone_data.mean(axis=0)
            # np.fill_diagonal(mean_corr, 0)  #-----ignore self-correlation----->
            mean_corr = (mean_corr-min(mean_corr.min(axis=1)))/(max(mean_corr.max(axis=1))-min(mean_corr.min(axis=1))) 

            # Plot heat‑map of the mean correlation matrix  
            # High-quality figure for scientific publication: Mean Correlation Matrix Heatmap
            plt.figure(figsize=(12, 10), dpi=300)
            ax = sns.heatmap(mean_corr,cmap="coolwarm",center=0,square=True,linewidths=0.7,linecolor='black',
                cbar_kws={"label": "Mean Correlation", "shrink": 0.85, "aspect": 30})
            plt.xlabel("Wavenumber (cm⁻¹)", fontsize=18, weight='bold')
            plt.ylabel("Wavenumber (cm⁻¹)", fontsize=18, weight='bold')
            plt.xticks(ticks=np.arange(len(row_label)) + 0.5,
                       labels=[int(label) for label in row_label],fontsize=14,rotation=90)
            plt.yticks(ticks=np.arange(len(row_label)) + 0.5,
                       labels=[int(label) for label in row_label],fontsize=14,rotation=0)
            plt.title(f"Mean Correlation Matrix (Zone {ii+1})", fontsize=20, weight='bold', pad=20)
            plt.gca().xaxis.set_label_position('top')
            plt.gca().xaxis.tick_top()
            plt.tight_layout()
            sns.despine()
            if self.save:
                plt.savefig(os.path.join(self.plot_dir, f'mean_correlation_matrix_{ii+1}.png'), bbox_inches='tight')
            # plt.show()


            #=========================Norm based symmetric and asymmetric feature extraction========================>   

            
            # Drop rows from mean_corr where the maximum value in the row is less than 0.2
            score_threshold = 0.2
            valid_rows = mean_corr.max(axis=1) >= score_threshold

            mean_corr = mean_corr[valid_rows][:, valid_rows]
            row_label_filtered = [label for i, label in enumerate(row_label) if valid_rows[i]]

            scale_mean_corr = mean_corr.copy()  # (mean_corr-min(mean_corr.min(axis=1)))/(max(mean_corr.max(axis=1))-min(mean_corr.min(axis=1)))  # Normalize the mean correlation matrix

            Asym_corr = (scale_mean_corr.T - scale_mean_corr)/scale_mean_corr.T # Skew‐symmetric part
            # Sym_corr = (scale_mean_corr + scale_mean_corr.T)/scale_mean_corr

            # per‐feature asymmetry scores
            Asym_score = np.linalg.norm(Asym_corr, axis=1)  # or axis=0, same result
            # Sym_score = np.linalg.norm(Sym_corr, axis=1)

            # find the index of the zone in the row_ind
            zone_val = list(set([int(i)for i in zone_var_med_ind[ii]]))    
            zone_ind = [row_ind.index(x) for x in zone_val]
            Asym_ind = [np.where(valid_rows[:x])[0].size for x in zone_ind]

            
            Asym_score = Asym_score-np.max(Asym_score[Asym_ind])
            Asym_cutoff = max(abs(Asym_score[Asym_ind]))

            # Create full-size arrays for asymmetry and symmetry scores
            full_Asym_score = np.full(len(row_label), np.nan)

            # Assign current scores to valid rows, rest remain NaN
            full_Asym_score[valid_rows] = Asym_score

            
            '''---------------Plotting------------'''

            # Plot full-size Asymmetry Scores per Feature (including NaNs for missing features)
            features = np.arange(len(full_Asym_score))  # Full feature indices
            plt.figure(figsize=(14, 8), dpi=300)
            bars = plt.bar(features, full_Asym_score,
                           color="#377eb8", edgecolor='black', width=0.7, linewidth=1.2)
            # Plot threshold lines for reference (only for valid features)
            plt.axhline(y=Asym_cutoff, color='#d7191c', linestyle='--', linewidth=2, label='Threshold')
            plt.axhline(y=-Asym_cutoff, color='#d7191c', linestyle='--', linewidth=2)
            plt.legend(fontsize=16, frameon=False)
            plt.xlabel('Wavenumber (cm⁻¹)', fontsize=20, weight='bold')
            plt.ylabel('Asymmetry Score', fontsize=20, weight='bold')
            plt.title('Full Asymmetry Scores per Feature', fontsize=22, weight='bold', pad=15)
            plt.xticks(ticks=np.arange(len(row_label)),
                       labels=[int(label) for label in row_label], fontsize=16, rotation=90)
            plt.yticks(fontsize=16)
            plt.grid(axis='y', linestyle='--', alpha=0.7, linewidth=1.2)
            plt.tight_layout()
            sns.despine()
            if self.save:
                plt.savefig(os.path.join(self.plot_dir, f'full_asymmetry_scores_{ii+1}.png'), bbox_inches='tight')
            # plt.show()

         
