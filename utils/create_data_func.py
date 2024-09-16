import numpy as np
import os
import h5py

DATADIR = '/home/ifernand/Cloud_SynthData_Proj/Antolik_Data/LSV1M_for_ML_MotionClouds/' # data directory of original data from Jan
WORKDIR = '/home/ifernand/Cloud_SynthData_Proj/'       # data directory for position and orientation info

def pull_data(robs_sq_rad, stim_dim, scale=None):

    if scale is not None:
        out_file = 'data/cloud_data_stim_dim'+str(stim_dim)+'_robs_sqrad_'+str(robs_sq_rad)+'_sca='+str(scale)+'.hdf5'
    else:
        out_file = 'data/cloud_data_stim_dim'+str(stim_dim)+'_robs_sqrad_'+str(robs_sq_rad)+'.hdf5'

    # check if file already exists
    check_file = os.path.isfile(out_file)
    if check_file:
        print('++WARNING: This file already exists!', out_file)
        return
    
    # List of all file names
    file_names = []
    for file_name in os.listdir(DATADIR):
        if file_name.startswith('motion'):
            file_scale = file_name[-5:-4]
            if scale is not None:
                if file_scale == str(scale):
                    file_names.append(file_name[:-4])
            else:
                file_names.append(file_name[:-4])

    N_files = len(file_names) # number of data files

    # STIMULUS
    # ========
    file_start_pos = [0] # starting position for each data file in time searies
    file_NT        = [] # number of time points in each file
    for i in range(N_files):
        stim_i = np.load(DATADIR+file_names[i]+'.npy')
        file_NT.append(stim_i.shape[0])

        # convert stim to int8 values
        scaled_stim = stim_i*255
        int8_stim = scaled_stim.astype(np.uint8)
        
        if i == 0:
            stim = int8_stim
            file_start_pos.append(stim.shape[0])
        elif i == N_files-1:
            stim = np.append(stim, int8_stim, axis=0)
        else:
            stim = np.append(stim, int8_stim, axis=0)
            file_start_pos.append(stim.shape[0])

    # crop stimulus
    center     = 110 # center of image
    sq_rad = stim_dim//2 # square radius of croped image
    pix_idx = (int(-1*sq_rad+center), int(sq_rad+center)) # indecies of image

    crop_stim = stim[:, pix_idx[0]:pix_idx[1], pix_idx[0]:pix_idx[1]]
    print('++INFO: stimuli of shape', crop_stim.shape)

    # flatten stim
    crop_flat_stim = crop_stim.reshape((crop_stim.shape[0], int(crop_stim.shape[1]*crop_stim.shape[2])))
    

    # POSITION AND ORINETATION
    # ========================
    pos_info = np.load(WORKDIR+'data/all_neuron_RF_centers_in_deg.pkl', allow_pickle=True) # RF position

    cell_type_list = ['X_ON', 'X_OFF', 'V1_Exc_L4', 'V1_Inh_L4', 'V1_Exc_L2/3', 'V1_Inh_L2/3'] # list of cell type in data set (LGN ON, V1 L 4, etc.)
    print('++INFO: Cell type order', cell_type_list)
    cell_key = []                     # list of cell keys in order
    cell_idx_dict = {}                # dictionary of cell type chosen indecies

    for i in range(len(cell_type_list)):
        cell_type  = cell_type_list[i]
        cell_x_pos = pos_info[cell_type][0,:]
        cell_y_pos = pos_info[cell_type][1,:]

        # chosse neurons based on position
        cell_x_pos_idx = np.where(np.logical_and(cell_x_pos>=-1*robs_sq_rad,cell_x_pos<=robs_sq_rad))[0] 
        cell_y_pos_idx = np.where(np.logical_and(cell_y_pos>=-1*robs_sq_rad,cell_y_pos<=robs_sq_rad))[0]
        
        # indecies for chosen cells
        cell_idx       = list(np.intersect1d(cell_x_pos_idx, cell_y_pos_idx))
        cell_NC        = len(cell_idx)
        print('++INFO:', cell_NC, cell_type, 'chosen')

        chosen_x_pos = cell_x_pos[cell_idx]
        chosen_y_pos = cell_y_pos[cell_idx]

        if i == 0:
            all_x_pos = chosen_x_pos
            all_y_pos = chosen_y_pos
        else:
            all_x_pos = np.append(all_x_pos, chosen_x_pos)
            all_y_pos = np.append(all_y_pos, chosen_y_pos)

        cell_key += [cell_type]*cell_NC
        cell_idx_dict[cell_type] = cell_idx

    # NEURON RESPONSES
    # ================
    for i in range(N_files):
        data = np.load(DATADIR+'responses_'+file_names[i]+'.pickle', allow_pickle=True)
        data_dict = list(data.values())[0]
        NT = file_NT[i]

        for j in range(len(cell_type_list)):
            cell_type = cell_type_list[j]
            cell_data = data_dict[cell_type]
            cell_idx  = cell_idx_dict[cell_type]
            cell_NC   = len(cell_idx)
            
            for k in range(cell_NC):
                spike_time = cell_data[0,cell_idx[k]]
                spike_count = np.histogram(spike_time, bins=NT, range=(0,int(16*NT)))[0].reshape(NT,1).astype(np.uint8)

                if j == 0 and k == 0:
                    cell_robs = spike_count
                else:
                    cell_robs = np.append(cell_robs, spike_count, axis=1)

        if i == 0:
            robs = cell_robs
        else:
            robs = np.append(robs, cell_robs, axis=0)
        print('++INFO: File', i, 'robs added')


    # CREATE FINAL FILE
    # =================
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('stim', data=crop_flat_stim)
        f.create_dataset('robs', data=robs)
        f.create_dataset('x_pos', data=all_x_pos)
        f.create_dataset('y_pos', data=all_y_pos)
        f.create_dataset('cell_key', data=cell_key)
        f.create_dataset('file_start_pos', data=file_start_pos)

    print('++INFO: Data file created:', out_file)


        
    


