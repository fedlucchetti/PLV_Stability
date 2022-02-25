import numpy as np
import sys, os, glob, csv
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

def plot_singlePLV(t,grand_average,plv,ton,toff,savg_size,savg_num,overlap):


    plv_noise = np.concatenate([plv[0:ton],plv[toff::]])
    fig, axs = plt.subplots(4)
    fig.suptitle('$size_{AVG}$ = '+str(savg_size) + "  $n_{AVG}$ = "+str(savg_num), fontsize=16)
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


    sig.RC8_V,sig.RC8_H           = utils.load_all_trials(files)
    print("sig.RC8_V.shape",sig.RC8_V.shape)
    # sys.exit()
    grand_average                 = sig.sub_average(savg_size=None,n_savg=1)
    plv_avg_matrix,plv_std_matrix = sig.plv_heatmap(savg_sizes,savg_nums,
                                                    SC_string="EFRV",phase_window_size=64)
    plv_avg_matrix                = sig.detect_outlier(plv_avg_matrix,plv_std_matrix,avg_std_ratio=3)
    ton,toff,freq                 = utils.ton,utils.toff,utils.freq_efr


    plot_heatmap(plv_avg_matrix,savg_sizes,savg_nums)
    # Select savg_size,savg_num region where PLV does not vary (approx)
    while True:
        try:
            savg_size = int(input(bcolors.WARNING + "Enter sAVG size [int]     : "))
            savg_num  = int(input(bcolors.WARNING + "Enter number of AVGs [int]: "))
        except Exception as e:
            print("Exception ",e,"unrecognized input")
            break
        plv       = sig.plv_single(savg_size,savg_num,SC_string="EFRV",phase_window_size=64)
        overlap=[]
        for savg_size in savg_sizes:
            overlap.append(1- 1/max(savg_size*50-N,1))
        plot_singlePLV(t,grand_average,plv,ton,toff,savg_size,savg_num,overlap)
        textinput = input("Try a different sAVG setting [y/n]?")
        if textinput=='y': continue
        elif textinput=='n': break
        else: print("Unrecognized: EXIT"); sys.exit();





if __name__ == '__main__':
    ###########################
    SAMPLEFOLDER = "samples"
    FFRFOLDER                     = join(utils.set_path(),"sAVG")
    files                         = np.array(utils.list_all_sAVG(join(FFRFOLDER)))
    # FFRFOLDER = join(SAMPLEFOLDER,"183551","ENV304_LE","85dB","sAVG")
    ############ Specify on off latencies ###############
    # ton=round(12/1000/DT);toff=round(69/1000/DT)
    ########################### Loop over the following sAVG settings
    savg_sizes   = np.arange(5,50,5)  # size of one sAVG
    savg_nums    = np.arange(50,100,5) # number of sAVGs
    ###########################
    main()
