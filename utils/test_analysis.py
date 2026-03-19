import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
#%%
def test_analysis(result_path, layers, scale, setting):
    """
    Args:
        result_path(string): Path to the result folder.
        layer_num(int): Layer number to analyze.
    """

    # Create a directory to save the plots
    plot_dir = os.path.join(result_path, setting, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    wavenumber =scale[0]
    time = scale[1]

    for layer_num in range(layers):
        print(f"Processing layer {layer_num + 1}/{layers}...")

        # Load the eigenvalues1 from the .npy file
        filename = os.path.join(result_path, setting, f'eigenvalues_layer_{layer_num+1}_testdata.npy')
        eigenvalues1 = np.load(os.path.abspath(filename))
        print(f"Loaded Activation from {filename}")

        # Check the shape of the eigenvalues1
        B, H, L = eigenvalues1.shape
        # eigenvalues1 = eigenvalues1.transpose(1, 2, 0)  # Reshape to (H, L, B)

        # Plot heatmaps for each H
        for h in range(H):
            plt.figure(figsize=(35, 20))
            eigenvalues_cleaned = np.nan_to_num(eigenvalues1[:, h, :], nan=1e17, posinf=1e17, neginf=-1e17) 
            
            eigenvalues_transposed = eigenvalues_cleaned.T
            
            plt.imshow(eigenvalues_transposed, aspect='auto', cmap='viridis')
            cbar = plt.colorbar()
            cbar.set_label('Magnitude', fontsize=20)
            cbar.ax.tick_params(labelsize=15)
            
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle='--', alpha=0.27)
            
            # Correct axis ticks
            nticks = [50, 50]
            plt.xticks(ticks=np.linspace(0, eigenvalues_cleaned.shape[0] - 1, nticks[0]), 
                   labels=np.round(np.linspace(time[0] / 60, time[-1] / 60, nticks[0]), 1), 
                   fontsize=15, rotation=90)
            plt.yticks(ticks=np.linspace(0, eigenvalues_transposed.shape[0] - 1, nticks[1]), 
                   labels=np.round(np.linspace(wavenumber[0], wavenumber[-1], nticks[1]), 0).astype(int), fontsize=15)
            
            # plt.title(f'Heatmap for L={layer_num+1} & H={h+1}', fontsize=17)
            plt.xlabel('Time (hr)', fontsize=24)
            plt.ylabel('Wavenumber (cm$^{-1}$)', fontsize=24)

            # Add y-axis on both ends
            ax = plt.gca()
            ax_secondary = ax.twinx()
            ax_secondary.set_yticks(np.linspace(0, eigenvalues_transposed.shape[0] - 1, nticks[1]))
            ax_secondary.set_yticklabels(np.round(np.linspace(wavenumber[0], wavenumber[-1], nticks[1]), 0).astype(int), fontsize=15)
            ax_secondary.tick_params(axis='y', labelsize=15)


            heatmap_filename = os.path.join(plot_dir, f'Heatmap_layer_{layer_num+1}_H_{h+1}.png')
            # plt.show()
            plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False)
            plt.close()
            print(f"Saved heatmap to {heatmap_filename}")

        # Plot concatenated heatmap for all H
        plt.figure(figsize=(35, 20))
        cleaned_eigenvalues = np.nan_to_num(eigenvalues1, nan=1e17, posinf=1e17, neginf=-1e17)
        concatenated_eigenvalues = (cleaned_eigenvalues - cleaned_eigenvalues.min(axis=1).min()) / (cleaned_eigenvalues.max(axis=1).max() -cleaned_eigenvalues.min(axis=1).min())#np.log(1+cleaned_eigenvalues)  # 

        concatenated_eigenvalues = np.array(concatenated_eigenvalues.sum(axis=1).T)

        # plt.imshow(concatenated_eigenvalues, aspect='auto', cmap='hot')#viridis

        masked_data = concatenated_eigenvalues#np.ma.masked_where(concatenated_eigenvalues<= 0.0000001, concatenated_eigenvalues)

        # Create a custom colormap: white to red
        cmap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "violet", "blue"])
        # cmap = plt.cm.viridis  # Use the 'inferno' colormap
        cmap.set_bad(color='white')  # Set masked values to white

        plt.imshow(masked_data, cmap=cmap,aspect='auto')
        cbar = plt.colorbar()
        cbar.set_label('Normalized Magnitude', fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.27)

        # Correct axis ticks
        nticks = [50, 50]
        plt.xticks(ticks=np.linspace(0, concatenated_eigenvalues.shape[1] - 1, nticks[0]),
               labels=np.round(np.linspace(time[0] / 60, time[-1] / 60, nticks[0]), 1),
               fontsize=15, rotation=90)
        plt.yticks(ticks=np.linspace(0, concatenated_eigenvalues.shape[0] - 1, nticks[1]),
               labels=np.round(np.linspace(wavenumber[0], wavenumber[-1], nticks[1]), 0).astype(int), fontsize=15)
        
        # plt.title(f'Concatenated Heatmap for L={layer_num+1}', fontsize=17)
        plt.xlabel('Time (hr)', fontsize=24)
        plt.ylabel('Wavenumber (cm$^{-1}$)', fontsize=24)

        # Add y-axis on both ends
        ax = plt.gca()
        ax_secondary = ax.twinx()
        ax_secondary.set_yticks(np.linspace(0, concatenated_eigenvalues.shape[0] - 1, nticks[1]))
        ax_secondary.set_yticklabels(np.round(np.linspace(wavenumber[0], wavenumber[-1], nticks[1]), 0).astype(int), fontsize=15)
        ax_secondary.tick_params(axis='y', labelsize=15)

        concatenated_heatmap_filename = os.path.join(plot_dir, f'Concatenated_Heatmap_layer_{layer_num+1}.png')
        # plt.show()
        plt.savefig(concatenated_heatmap_filename, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False)
        plt.close()
        print(f"Saved concatenated heatmap to {concatenated_heatmap_filename}")


        for h in range(H):
            eigenvalues_cleaned = np.nan_to_num(eigenvalues1[:, h, :], nan=1e17, posinf=1e17, neginf=-1e17)
            rows, cols = np.nonzero(eigenvalues_cleaned)
            unique_cols, counts = np.unique(cols, return_counts=True)
            total_counts= 5
            valid_cols = unique_cols[counts >= total_counts]  # Keep columns with at least 5 consecutive non-zero values
            # mask = np.isin(cols, valid_cols)
            # rows, cols = rows[mask], cols[mask]
    
            # Find the top `top_n` values for each row and their column indices
            top_n = min(len(valid_cols),10)

            top_values = []
            top_indices = []
            for row in range(eigenvalues_cleaned.shape[0]):
                row_data = eigenvalues_cleaned[row, :]
                top_indices_row = row_data.argsort()[-top_n:][::-1]  # Indices of top `top_n` values
                top_values_row = row_data[top_indices_row]  # Top `top_n` values
                top_indices_row = np.where(top_values_row > 0.0, top_indices_row, np.nan)

                top_indices.append(top_indices_row)
                top_values.append(top_values_row)

            top_indices = np.array(top_indices)  # No need to transpose as rows are already aligned
            top_values = np.array(top_values)

            # Plot the top `top_n` values for each row
            plt.figure(figsize=(25, 15))
            for i in range(top_n):
                plt.scatter(range(eigenvalues_cleaned.shape[0]), top_indices[:, i], label=f'Top {i+1}', s=10, alpha=0.7, marker='o')
            
            plt.xlabel('Time (hr)', fontsize=17)
            plt.ylabel('Wavenumber (cm$^{-1}$)', fontsize=17)
            
            nticks = [50, 50]
            plt.xticks(ticks=np.linspace(0, concatenated_eigenvalues.shape[1] - 1, nticks[0]), 
                   labels=np.round(np.linspace(time[0] / 60, time[-1] / 60, nticks[0]), 1), 
                   fontsize=15, rotation=90)
            plt.yticks(ticks=np.linspace(0, concatenated_eigenvalues.shape[0] - 1, nticks[1]), 
                   labels=np.round(np.linspace(wavenumber[0], wavenumber[-1], nticks[1]), 0).astype(int), fontsize=15)

            plt.title(f'Top {top_n} Activation per Row for Layer {layer_num+1}, H={h+1}', fontsize=17)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.27)    

            top_values_filename = os.path.join(plot_dir, f'top{top_n}_values_layer_{layer_num+1}_H_{h+1}_rows.png')
            plt.savefig(top_values_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved top {top_n} values plot to {top_values_filename}")

        # Generate top-n plot for concatenated H data
        plt.figure(figsize=(25, 15))
        rows, cols = np.nonzero(concatenated_eigenvalues)
        unique_cols, counts = np.unique(cols, return_counts=True)
        total_counts= 3
        valid_cols = unique_cols[counts >= total_counts] # Keep columns with at least 5 consecutive non-zero values
        # mask = np.isin(cols, valid_cols)
        # rows, cols = rows[mask], cols[mask]

        top_n = min(len(valid_cols),20)
        concatenated_top_values = []
        concatenated_top_indices = []
        concatenated_data = concatenated_eigenvalues.T
        for row in range(concatenated_data.shape[0]):
            row_data = concatenated_data[row, :]
            top_indices_row = row_data.argsort()[-top_n:][::-1]  # Indices of top `top_n` values
            top_values_row = row_data[top_indices_row]  # Top `top_n` values
            top_indices_row = np.where(top_values_row > 0.0, top_indices_row, np.nan)

            concatenated_top_indices.append(top_indices_row)
            concatenated_top_values.append(top_values_row)

        concatenated_top_indices = np.array(concatenated_top_indices)
        concatenated_top_values = np.array(concatenated_top_values)

        for i in range(top_n):
            plt.scatter(range(concatenated_data.shape[0]), concatenated_top_indices[:, i], label=f'Top {i+1}', s=10, alpha=0.7, marker='o')

        plt.xlabel('Time (hr)', fontsize=17)
        plt.ylabel('Wavenumber (cm$^{-1}$)', fontsize=17)
        
        nticks = [50, 50]
        plt.xticks(ticks=np.linspace(0, concatenated_data.shape[0] - 1, nticks[0]), 
                labels=np.round(np.linspace(time[0] / 60, time[-1] / 60, nticks[0]), 1), 
                fontsize=15, rotation=90)
        plt.yticks(ticks=np.linspace(0, concatenated_data.shape[1] - 1, nticks[1]), 
                labels=np.round(np.linspace(wavenumber[0], wavenumber[-1], nticks[1]), 0).astype(int), fontsize=15)

        plt.title(f'Top {top_n} Activation per Row for Concatenated H Data, Layer {layer_num+1}', fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.27)  

        concatenated_top_values_filename = os.path.join(plot_dir, f'top{top_n}_values_layer_{layer_num+1}_concatenated_H_rows.png')
        plt.savefig(concatenated_top_values_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved top {top_n} values plot for concatenated H data to {concatenated_top_values_filename}")


    #%%% Over all plot across layers
    # Generate a concatenated heatmap across all H and all layers

    filename = os.path.join(result_path, setting, f'eigenvalues_layer_{1}_testdata.npy')
    eigenvalues1 = np.load(os.path.abspath(filename))
    cleaned_eigenvalues = np.nan_to_num(eigenvalues1, nan=1e17, posinf=1e17, neginf=-1e17)
    overall_data = cleaned_eigenvalues.sum(axis=1)  # Sum across H
    overall_data = (overall_data - overall_data.min(axis=1).min()) / (overall_data.max(axis=1).max() - overall_data.min(axis=1).min())

    for layer_num in range(1,layers):
        filename = os.path.join(result_path, setting, f'eigenvalues_layer_{layer_num+1}_testdata.npy')
        eigenvalues1 = np.load(os.path.abspath(filename))
        cleaned_eigenvalues = np.nan_to_num(eigenvalues1, nan=1e17, posinf=1e17, neginf=-1e17)
        concatenated_eigenvalues = cleaned_eigenvalues.sum(axis=1)  # Sum across H
        concatenated_eigenvalues = (concatenated_eigenvalues - concatenated_eigenvalues.min(axis=1).min()) / (concatenated_eigenvalues.max(axis=1).max() - concatenated_eigenvalues.min(axis=1).min())
        overall_data = overall_data + concatenated_eigenvalues  # Sum across layers

    # Normalize the overall data
    # overall_data = (overall_data - overall_data.min(axis=1).min()) / (overall_data.min(axis=1).max() - overall_data.min(axis=1).min())
    

    # Plot the overall heatmap
    plt.figure(figsize=(35, 20))
    overall_data = np.array(overall_data)  # Ensure overall_data is a NumPy array
    overall_data = overall_data#np.ma.masked_where(overall_data <= 0.001, overall_data)
    # Create a custom colormap: white to red
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "violet", "blue"])
    # cmap = plt.cm.viridis  # Use the 'inferno' colormap
    cmap.set_bad(color='white')  # Set masked values to white
    
    plt.imshow(overall_data.T, aspect='auto', cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label('Normalized Magnitude', fontsize=20)
    cbar.ax.tick_params(labelsize=15)

    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.27)

    # Correct axis ticks
    nticks = [50, 50]
    plt.xticks(ticks=np.linspace(0, overall_data.shape[0] - 1, nticks[0]),
            labels=np.round(np.linspace(time[0] / 60, time[-1] / 60, nticks[0]), 1),
            fontsize=15, rotation=90)
    plt.yticks(ticks=np.linspace(0, overall_data.shape[1] - 1, nticks[1]),
            labels=np.round(np.linspace(wavenumber[0], wavenumber[-1], nticks[1]), 0).astype(int), fontsize=15)

    # plt.title('Overall Heatmap Across All Layers and H', fontsize=17)
    plt.xlabel('Time (hr)', fontsize=24)
    plt.ylabel('Wavenumber (cm$^{-1}$)', fontsize=24)

    # Add y-axis on both ends
    ax = plt.gca()
    ax_secondary = ax.twinx()
    ax_secondary.set_yticks(np.linspace(0, overall_data.shape[1] - 1, nticks[1]))
    ax_secondary.set_yticklabels(np.round(np.linspace(wavenumber[0], wavenumber[-1], nticks[1]), 0).astype(int), fontsize=15)
    ax_secondary.tick_params(axis='y', labelsize=15)

    overall_heatmap_filename = os.path.join(plot_dir, 'Overall_Heatmap_All_Layers_H.png')
    plt.savefig(overall_heatmap_filename, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.close()
    print(f"Saved overall heatmap to {overall_heatmap_filename}")
    

    print("Analysis complete.")
