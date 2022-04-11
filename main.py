import numpy as np
import sys, os, glob, csv, json
from os.path import join, split
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sp
from threading import Thread


###########################
from bin import utils, signaltools, bcolors
utils   = utils.Utils()
sig     = signaltools.SignalTools(utils)
bcolors = bcolors.bcolors()
###########################
FS   = 24414
DT   = 1/FS
N    = 2048
t    = np.arange(0,DT*N,DT)
freq = np.arange(0,FS/2,FS/N)
###########################
FONTSIZE=16
########################### MAIN ###########################

def plot_heatmap(plv_avg_matrix,savg_sizes,savg_nums):
    # Plot the surface.
    X, Y = np.meshgrid(savg_sizes, savg_nums)
    plt.pcolormesh(X, Y, plv_avg_matrix,cmap ='hot', vmin = 0.0, vmax = 1.0)
    plt.xlabel("AVG size ",fontsize=FONTSIZE)
    plt.ylabel("$n_{AVG}$ ",fontsize=FONTSIZE)
    cbar = plt.colorbar()
    cbar.set_label('PLV',fontsize=FONTSIZE)
    try:
        cbar.set_ticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=FONTSIZE)
    except:
        cbar.set_ticklabels([0,0.2,0.4,0.6,0.8,1])
    plt.show(block=0)

def plot_plv_overlap_map(plv_avg_matrix,overlap_array,savg_nums,Ntot):
    for id,savg_num in enumerate(savg_nums):
        plt.plot(overlap_array, plv_avg_matrix[id],label="$N_{AVG}$="+str(savg_num))
        plt.xlabel("Overlap index")
        plt.ylabel("PLV")
    plt.legend()
    plt.show(block=0)

def plot_singlePLV(t,grand_average,plv,ton,toff,savg_num,overlap):
    savg_size = np.sqrt(overlap/savg_num) * sig.n_waveforms
    plv_noise = np.concatenate([plv[0:ton],plv[toff::]])
    fig, axs  = plt.subplots(4)
    fig.suptitle('$size_{AVG}$ = '+str(savg_size) + "  $n_{AVG}$ = "+str(savg_num)+" overlap idx "+str(overlap), fontsize=16)
    axs[0].plot(t,grand_average[0]["V"][:,0],"b")
    axs[1].plot(t[ton:toff],plv[ton:toff],"r")
    axs[1].plot(t[0:ton],plv[0:ton]      ,"k")
    axs[1].plot(t[toff::],plv[toff::]    ,"k")
    axs[1].set(xlabel="Time [s]", ylabel='PLV')
    label = "PLV = " + str(round( plv[ton:toff].mean(),2)) + "+-" + str(round( plv[ton:toff].std(),2))
    axs[3].hist(plv[ton:toff],label=label,color="r")
    axs[3].hist(plv_noise    ,label="noise floor",color="k")
    axs[3].set(xlabel="PLV"  ,ylabel='counts')
    axs[3].legend()
    plt.show(block=0)

def main():
    SC                                              = utils.SC
    channel                                         = utils.channel
    sig.RC8_V,sig.RC8_H,sig.RCnoise_V,sig.RCnoise_H = utils.load_all_trials(files)
    sig.n_waveforms                                 = len(files)*16
    print("sig.RC8_V.shape",sig.RC8_V.shape)
    grand_average                                   = sig.sub_average(savg_size=None,n_savg=1)
    # plv_avg_matrix,plv_std_matrix = sig.plv_heatmap(savg_sizes,savg_nums,
    #                                                 SC_string="EFRV",phase_window_size=64)
    plv_avg_matrix,plv_std_matrix                   = sig.plv_heatmap_overlap(overlap_array,savg_nums,
                                                                                SC_string=SC+channel,phase_window_size=64)
    plv_avg_matrix                                  = sig.detect_outlier(plv_avg_matrix,plv_std_matrix,avg_std_ratio=3)
    

    # # plot_heatmap(plv_avg_matrix,savg_sizes,savg_nums)
    # plot_plv_overlap_map(plv_avg_matrix,overlap_array,savg_nums,sig.n_waveforms)
    # # Select savg_size,savg_num region where PLV does not vary (approx)
    # while True:
    #     try:
    #         overlap   = float(input(bcolors.WARNING + "Enter overlap index      : "))
    #         savg_num  = int(input(bcolors.WARNING + "Enter number of AVGs [50 default]: "))
    #     except Exception as e:
    #         print("Exception ",e,"unrecognized input")
    #         continue
    #     plv       = sig.plv_single(None,savg_num,SC_string="EFRV",phase_window_size=64,overlap=overlap)
    #     plot_singlePLV(t,grand_average,plv,ton,toff,savg_num,overlap)
    #     textinput = input("Try a different sAVG setting [y/n]?")
    #     if textinput=='y': continue
    #     elif textinput=='n': break
    #     else: print("Unrecognized: EXIT"); sys.exit();

    savg_num  = 50
    std_plv = np.std(plv_avg_matrix,axis=0)
    for e in sorted(std_plv):
        o_i = overlap_array[np.where(std_plv==e)[0][0]]
        if o_i > 0.1 and o_i < 6:
            overlap = o_i
            break
        else:
            overlap = 1.4
            continue
    
    for ch in ["V","H"]:
        ton,toff       = utils.ton[ch],utils.toff[ch]
        plv, plv_noise = sig.plv_single(None,savg_num,SC_string=SC+ch,phase_window_size=64,overlap=overlap)

        ## Save PLV in .json
        with open(utils.Meta_AVG_data_path) as data_file: data = json.load(data_file)
        plv_dist, plv_noise_dist = [],[]
        for i,_plv in enumerate(plv[ton:toff]):
            plv_dist.append(str(round(_plv,3)))
            plv_noise_dist.append(str(round(plv_noise[ton:toff][i],3)))
        data["FFR"]["Channel-"+ch][SC]["Analysis"]['PLV'] = ",".join(plv_dist)
        data["FFR"]["Channel-"+ch]["Noise"]["Analysis"]['PLV'] = ",".join(plv_noise_dist)
        with open(utils.Meta_AVG_data_path, 'w') as outfile: json.dump(data, outfile,ensure_ascii=False)




if __name__ == '__main__':
    ###########################
    SAMPLEFOLDER = "samples"
    FFRFOLDER                     = join(utils.set_path(),"sAVG")
    files                         = np.array(utils.list_all_sAVG(join(FFRFOLDER)))

    # FFRFOLDER = join(SAMPLEFOLDER,"183551","ENV304_LE","85dB","sAVG")
    ############ Specify on off latencies ###############
    # ton=round(12/1000/DT);toff=round(69/1000/DT)
    ########################### Loop over the following sAVG settings
    savg_sizes    = np.arange(100,1100,100)  # size of one sAVG
    savg_nums     = np.arange(50,110,10) # number of sAVGs
    overlap_array = np.arange(0.1,10,0.5)
    ###########################
    main()
