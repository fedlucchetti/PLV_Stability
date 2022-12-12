import scipy as sp
import numpy as np
import sys,os,glob, csv
from os.path import join, split
from scipy.signal import hilbert
from tqdm import tqdm
from multiprocessing import Process, Manager
from threading import Thread, Event, Lock


from bin import bcolors
bcolors=bcolors.bcolors()
# utils = utils.Utils()

##################################3
N    = 2048
FS   = 24414
DT   = 1/FS
N    = 2048
t    = np.arange(0,DT*N,DT)
##################################3
##################################3


class SignalTools(object):
    def __init__(self,utils):
        self.utils = utils
        self.rotation_matrix = self.utils.rotation_matrix()
        self.RC8_V,self.RC8_H = None,None
        self.RCnoise_V,self.RCnoise_H = None,None
    def fft(self,waveform):
        return np.abs(np.fft.fft(waveform))

    def __prepare_R8(self,waveforms_RC):
        """
        transform the 16 RC waveforms from the Labview/TdT buffer into the 8 Ri waveforms
        before gPTPV rotation
        """
        waveforms_R8      = np.zeros([N,8])
        waveforms_R8[:,0] = (waveforms_RC[:,0]+np.sum(waveforms_RC[:,4:8],axis=1))/5.0
        waveforms_R8[:,1] = waveforms_RC[:,1]
        waveforms_R8[:,2] = waveforms_RC[:,2]
        waveforms_R8[:,3] = waveforms_RC[:,3]
        waveforms_R8[:,4] = (waveforms_RC[:,8]+np.sum(waveforms_RC[:,12:16],axis=1))/5.0
        waveforms_R8[:,5] = waveforms_RC[:,9]
        waveforms_R8[:,6] = waveforms_RC[:,10]
        waveforms_R8[:,7] = waveforms_RC[:,11]
        return waveforms_R8

    def gptpv(self,waveforms_RC):
        """
        applies gPTPV matrix rotation on R8 waveforms_sAVGs_H
        """
        waveforms_R8         = self.__prepare_R8(waveforms_RC) # see Labview implementation
        waveforms_analytical = np.transpose(sp.signal.hilbert(waveforms_R8))
        waveforms_gptpv      = np.matmul(np.real(self.rotation_matrix),np.real(waveforms_analytical))
        waveforms_gptpv      = waveforms_gptpv - np.matmul(np.imag(self.rotation_matrix),np.imag(waveforms_analytical))
        return waveforms_gptpv/8.0

    def average(self,RC_indices):
        """
        Averages over all specified indiced
        """
        waveforms_RC_V,waveforms_RC_H = np.zeros([N,16]),np.zeros([N,16])
        noise_RC_V,noise_RC_H = np.zeros([N,16]),np.zeros([N,16])
        for RC_index in RC_indices:
            waveforms_RC_V+=self.RC8_V[RC_index]
            waveforms_RC_H+=self.RC8_H[RC_index]
            noise_RC_V+=self.RCnoise_V[RC_index]
            noise_RC_H+=self.RCnoise_H[RC_index]
        waveforms_RC_V,waveforms_RC_H=waveforms_RC_V/len(RC_indices),waveforms_RC_H/len(RC_indices)
        noise_RC_V,noise_RC_H=noise_RC_V/len(RC_indices),noise_RC_H/len(RC_indices)
        waveforms_gptpv_V = np.transpose(self.gptpv(waveforms_RC_V))
        waveforms_gptpv_H = np.transpose(self.gptpv(waveforms_RC_H))
        noise_V,noise_H = np.sum(noise_RC_V,axis=1)/16,np.sum(noise_RC_H,axis=1)/16

        return waveforms_gptpv_V,waveforms_gptpv_H,noise_V,noise_H

    def sub_average(self,savg_size,n_savg):
        """
        Computes n_savg subaverages computed over savg_size (= number of trials per sAVG)
        ------------------------------------------------------------------------------------------
        savg_sizes        : size of a sub average (=number of trials)
        savg_nums         : number of subaverages
        ------------------------------------------------------------------------------------------
        """

        n_workers         = n_savg
        list_file_id      = np.arange(0,self.RC8_V.shape[0])
        shared_dict       = Manager().dict()
        for id in range(n_savg): shared_dict.update({id:{"V":np.zeros([N,8]),"H":np.zeros([N,8]),"noiseV":np.zeros(N),"noiseH":np.zeros(N)}})
        if n_savg==1:
            waveforms_sAVGs_V,waveforms_sAVGs_H,noise_sAVGs_V,noise_sAVGs_H  = self.average(list_file_id)
            shared_dict.update({0:{"V":waveforms_sAVGs_V,"H":waveforms_sAVGs_H,'noiseV':noise_sAVGs_V,'noiseH':noise_sAVGs_H}})
            return shared_dict
        else:
            def worker(shared_dict,worker_id):
                np.random.shuffle(list_file_id)
                waveforms_sAVGs_V,waveforms_sAVGs_H,noise_sAVGs_V,noise_sAVGs_H = self.average(list_file_id[0:savg_size])
                shared_dict.update({worker_id:{"V":waveforms_sAVGs_V,"H":waveforms_sAVGs_H,'noiseV':noise_sAVGs_V,'noiseH':noise_sAVGs_H}})
            threads=list()
            for worker_id in range(n_workers): threads.append(Thread(target=worker, args=(shared_dict,worker_id)))
            for thread in threads:             thread.start()
            return shared_dict



    def plv_heatmap(self,savg_sizes,savg_nums,SC_string="EFRV",phase_window_size=50):
        """
        computes the plv heatmap over varying number of subaverages of different sizes
        ------------------------------------------------------------------------------------------
        savg_sizes        : size of a sub average (=number of trials) [array]
        savg_nums         : number of subaverages [array]
        SC_string         : EFRV, CDTV, F1V, F2V, EFRH, CDTH, F1H, F2H
        phase_window_size : window size over which a local phase average is computed
        ------------------------------------------------------------------------------------------
        """
        ton,toff = self.utils.ton[SC_string[-1]], self.utils.toff[SC_string[-1]]
        print(bcolors.HEADER,"\n START: \t" + bcolors.ENDC + "Multithreading PLV matrix computation")
        shared_dict         = Manager().dict()
        plv_avg_matrix      = np.zeros([len(savg_nums),len(savg_sizes)     ])
        plv_std_matrix      = np.zeros([len(savg_nums),len(savg_sizes)     ])
        waveform_idx        = self.__get_waveform_idx(SC_string)
        n_workers           = len(savg_nums)

        def worker(self,plv_avg_matrix,plv_std_matrix,id1):
            for id2,savg_size in enumerate(tqdm(savg_sizes)):
                plv,_                     = self.plv_single(savg_size,savg_nums[id1], SC_string=SC_string,
                                                          phase_window_size=phase_window_size)
                plv_avg_matrix[id1,id2] = np.mean(plv[ton:toff])
                plv_std_matrix[id1,id2] = np.std(plv[ton:toff])

        threads=list()
        for worker_id in range(n_workers):
            threads.append(Thread(target=worker, args=(self,plv_avg_matrix,plv_std_matrix,worker_id)))
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        print(bcolors.OKGREEN,"\n DONE: multithreading PLV matric computation \n")
        # os.system("clear")

        return plv_avg_matrix,plv_std_matrix

    def plv_heatmap_overlap(self,overlap_array,savg_nums,SC_string="EFRV",phase_window_size=50):
        """
        computes the plv heatmap over varying number of subaverages of different sizes
        ------------------------------------------------------------------------------------------
        savg_sizes        : size of a sub average (=number of trials) [array]
        savg_nums         : number of subaverages [array]
        SC_string         : EFRV, CDTV, F1V, F2V, EFRH, CDTH, F1H, F2H
        phase_window_size : window size over which a local phase average is computed
        ------------------------------------------------------------------------------------------
        """
        ton,toff = self.utils.ton[SC_string[-1]], self.utils.toff[SC_string[-1]]
        print(bcolors.HEADER,"\n START: \t" + bcolors.ENDC + "Multithreading PLV matrix computation")
        shared_dict         = Manager().dict()
        plv_avg_matrix      = np.zeros([len(savg_nums),len(overlap_array)     ])
        plv_std_matrix      = np.zeros([len(savg_nums),len(overlap_array)     ])
        waveform_idx        = self.__get_waveform_idx(SC_string)
        n_workers           = len(savg_nums)

        def worker(self,plv_avg_matrix,plv_std_matrix,id1):
            for id2,overlap in enumerate(tqdm(overlap_array)):
                plv,_                     = self.plv_single(0,savg_nums[id1], SC_string=SC_string,
                                                          phase_window_size=phase_window_size,overlap=overlap)
                plv_avg_matrix[id1,id2] = np.mean(plv[ton:toff])
                plv_std_matrix[id1,id2] = np.std(plv[ton:toff])

        threads=list()
        for worker_id in range(n_workers):
            threads.append(Thread(target=worker, args=(self,plv_avg_matrix,plv_std_matrix,worker_id)))
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        print(bcolors.OKGREEN,"\n DONE: multithreading PLV matric computation \n")
        # os.system("clear")

        return plv_avg_matrix,plv_std_matrix

    def plv_single(self,savg_size,n_savg,SC_string="EFRV",phase_window_size=50,overlap=-1):
        """
        Computes PLV for a given number of subaverages of fixed size
        ------------------------------------------------------------------------------------------
        savg_size         : size of a sub average (=number of trials)
        savg_num          : number of subaverages
        SC_string         : EFRV, CDTV, F1V, F2V, EFRH, CDTH, F1H, F2H
        phase_window_size : window size over which a local phase average is computed
        freq              : waveform frequency
        ------------------------------------------------------------------------------------------
        """
        if overlap!=-1: savg_size = round(np.sqrt(overlap/n_savg) * self.n_waveforms/16)
        # print(overlap,savg_size)
        subAVG_dict  = self.sub_average(savg_size=savg_size,n_savg=n_savg)
        phase_sc,phase_noise    = np.zeros([n_savg,N]),np.zeros([n_savg,N])
        waveform_idx = self.__get_waveform_idx(SC_string)
        for id_avg in subAVG_dict:
            analytic_signal     = hilbert(subAVG_dict[id_avg][SC_string[-1]][:,waveform_idx]) # [savg id ][channel][:,waveform_idx]
            phase_sc[id_avg]   = np.abs(np.unwrap(np.angle(analytic_signal)) - 2*np.pi*self.utils.freq_SC[SC_string[:-1]]*t)
            if SC_string[:-1]=='EFR': #To avoid calculating noise PLV for each SC PLV calculation
                an_sig_noise       = hilbert(subAVG_dict[id_avg]["noise"+SC_string[-1]])
                phase_noise[id_avg] = np.unwrap(np.angle(an_sig_noise))
        plv, plv_noise = np.zeros([N]),np.zeros([N])
        for it in range(N):
            it_min,it_max       = max(0,it-round(phase_window_size/2)),min(it+round(phase_window_size/2),N)
            plv[it]             = np.abs(np.mean(np.exp(1j*phase_sc[:,it_min:it_max])))
            if SC_string[:-1]=='EFR':
                plv_noise[it]       = np.abs(np.mean(np.exp(1j*phase_noise[:,it_min:it_max])))
        
        if SC_string[:-1]=='EFR':
            return plv, plv_noise
        else:
            return plv


    def detect_outlier(self,plv_avg_matrix,plv_std_matrix,avg_std_ratio=3):
        """
        sets plv_avg_matrix extries to 0 if std to big (avg < avg_std_ratio * std)
        """
        for id1  in range(plv_avg_matrix.shape[0]):
            for id2 in range(plv_avg_matrix.shape[1]):
                if plv_avg_matrix[id1,id2] < avg_std_ratio*plv_std_matrix[id1,id2]:
                    plv_avg_matrix[id1,id2]=0.0
        return plv_avg_matrix

    def __get_waveform_idx(self,SC_string):
        if  SC_string[:-1] == 'EFR': return 0
        elif  SC_string[:-1] == 'EFR**': return 1
        elif  SC_string[:-1] == 'EFR***': return 2
        elif  SC_string[:-1] == 'F1': return 3
        elif  SC_string[:-1] == 'F2': return 4
        elif  SC_string[:-1] == 'CDT': return 5
        elif  SC_string[:-1] == 'CDT*': return 6
        else:
            print(bcolors.FAIL,"Unrecognized SC string")
            sys.exit()
