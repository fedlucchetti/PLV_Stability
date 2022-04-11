import numpy as np
import sys,os,glob, csv, json
from os.path import join, split
from tqdm import tqdm
from tkinter import filedialog as fd
from threading import Thread


from bin import bcolors
bcolors=bcolors.bcolors()
N = 2048; DT = 1/24414
sAVG_FILE_PATTERN = "*xls"
DATAFOLDER = "data"


class Utils:
    def __init__(self):
        self.GPTPV_FOLDER = join(DATAFOLDER,"gptpv")

    def rotation_matrix(self):
        with open(join(self.GPTPV_FOLDER,"real_sol.txt")) as file:
            gptpv_real = np.array([[float(digit) for digit in line.split("\t")] for line in file])
        with open(join(self.GPTPV_FOLDER,"imag_sol.txt")) as file:
            gptpv_imag = np.array([[float(digit) for digit in line.split("\t")] for line in file])
        return gptpv_real + 1j * gptpv_imag


    def list_all_sAVG(self,root_path):
         return glob.glob(join(root_path,sAVG_FILE_PATTERN))

    def open_sAVG_file(self,path):
        data= []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t')
            for id,row in enumerate(spamreader):
                if id >1:
                    for id, entry in enumerate(row):
                        row[id]=float(entry.replace(",","."))
                    data.append(row)
        return np.array(data)

    def set_path(self):
        # path = fd.askopenfilenames(title='Choose a Meta_AVG_data file',
        #                            filetypes=[('all Meta_AVG_data files', '*Meta_AVG_data.json')])
        # self.Meta_AVG_data_path = path[0]
        # self.get_ton_toff_freq()
        # return os.path.split(path[0])[0]
        ## Read from cmd: python main.py (path to Meta_AVG_data.json) (SC to analyze)
        try:
            self.Meta_AVG_data_path = sys.argv[1]
            self.SC = sys.argv[2]
            self.get_ton_toff_freq()
        except:
            print("Input error \nEnter: python main.py (path to Meta_AVG_data.json) (SC to analyze)"); sys.exit();
        
        return os.path.dirname(self.Meta_AVG_data_path)

    def get_ton_toff_freq(self):
        print("opening ",self.Meta_AVG_data_path)
        with open(self.Meta_AVG_data_path) as data_file: data = json.load(data_file)
        self.ton     = {"V":0,"H":0}
        self.toff    = {"V":N,"H":N}
        
        f1           = data["MetaData"]["Stimulus"]["F1"]
        f2           = data["MetaData"]["Stimulus"]["F2"]
        frequency    = {'EFR':f2-f1,'EFR**':2*(f2-f1),'EFR***':3*(f2-f1),'F1':f1,'F2':f2,'CDT':2*f1-f2,'CDT*':2*f2-f1}
        self.freq_SC = frequency[self.SC]

        for ch in ["V","H"]:
            latency = data["FFR"]["Channel-"+ch][self.SC]["Analysis"]["Latency"]
            length  = data["FFR"]["Channel-"+ch][self.SC]["Analysis"]["Lenght"]
            if latency!='' and type(latency)==str:
                self.ton[ch]  =                round(float(latency.replace(",","."))/1000.0/DT)
                self.toff[ch] = self.ton[ch] + round(float(length.replace(",","."))/1000.0/DT)
                self.channel  = ch 
            elif latency!=-1 and  type(latency)==float:
                self.ton[ch]  =                round(latency/1000.0/DT)
                self.toff[ch] = self.ton[ch] + round(length/1000.0/DT)
                self.channel  = ch 
            else:
                print("SC non-detected in Channel V nor H"); sys.exit();
        
        self.Noise = {"Channel-V":data["FFR"]["Channel-V"]["Noise"]["AVG"]["Waveform"],"Channel-H":data["FFR"]["Channel-H"]["Noise"]["AVG"]["Waveform"]}


    def overlap(self, N, sizes):
        return 6*sizes**2*(N**2-N)

    def load_all_trials(self,files):

        n_workers  = 10
        n_files    = len(files)
        n_subfiles = round(n_files/n_workers)
        if len(files)==0:
            print(bcolors.FAIL,"No sAVG files found, EXIT")
            sys.exit()
        print("\n",bcolors.HEADER,"START:",bcolors.ENDC," Loading all trials into memory")
        RC8_V,RC8_H      = np.zeros([n_files,N,16]),np.zeros([n_files,N,16])
        RCnoise_V,RCnoise_H = np.zeros([n_files,N,16]),np.zeros([n_files,N,16])
        file_indices     = np.arange(0,n_files)
        sub_files        = [files[i:i + n_subfiles] for i in range(0, len(files), n_subfiles)]
        sub_file_indices = [file_indices[i:i + n_subfiles] for i in range(0, len(files), n_subfiles)]

        def worker(RC8_V,RC8_H,RCnoise_V,RCnoise_H,worker_id):
            _file_indices = sub_file_indices[worker_id]
            for file_index in tqdm(_file_indices):
                data                                = self.open_sAVG_file(files[file_index])
                RC8_V[file_index],RC8_H[file_index],RCnoise_V[file_index],RCnoise_H[file_index] = self.format_to_R_waveforms(data)

        threads=[]
        for worker_id in range(n_workers): threads.append(Thread(target=worker, args=(RC8_V,RC8_H,RCnoise_V,RCnoise_H,worker_id)))
        for thread in threads:             thread.start()
        for thread in threads:             thread.join()

        shuffle_idx  = np.arange(RC8_V.shape[0])
        new_H, new_V = np.zeros(RC8_V.shape),np.zeros(RC8_H.shape)
        new_noise_H, new_noise_V = np.zeros(RCnoise_H.shape),np.zeros(RCnoise_V.shape)
        for r_id in range(RC8_V.shape[-1]):
            np.random.shuffle(shuffle_idx)
            new_V[:,:,r_id],new_H[:,:,r_id] = RC8_V[shuffle_idx,:,r_id],RC8_H[shuffle_idx,:,r_id]
            new_noise_V[:,:,r_id],new_noise_H[:,:,r_id] = RCnoise_V[shuffle_idx,:,r_id],RCnoise_H[shuffle_idx,:,r_id]
        print("\n",bcolors.OKGREEN,"DONE:",bcolors.ENDC,"  Loading all trials into memory \n \n \n",)
        return new_V,new_H, new_noise_V, new_noise_H

    def format_to_R_waveforms(self,data):
        waveforms_RC_V = np.zeros([N,16])
        waveforms_RC_H = np.zeros([N,16])
        noise_RC_V = np.zeros([N,16])
        noise_RC_H = np.zeros([N,16])
        for block_id in range(8):
            waveforms_RC_V[:,block_id]   = data[block_id*N:(block_id+1)*N,0]
            waveforms_RC_V[:,8+block_id] = data[block_id*N:(block_id+1)*N,2]
            waveforms_RC_H[:,block_id]   = data[block_id*N:(block_id+1)*N,4]
            waveforms_RC_H[:,8+block_id] = data[block_id*N:(block_id+1)*N,6]

            noise_RC_V[:,block_id]       = data[block_id*N:(block_id+1)*N,1]
            noise_RC_V[:,8+block_id]     = data[block_id*N:(block_id+1)*N,3]
            noise_RC_H[:,block_id]       = data[block_id*N:(block_id+1)*N,5]
            noise_RC_H[:,8+block_id]     = data[block_id*N:(block_id+1)*N,7]
        return waveforms_RC_V,waveforms_RC_H,noise_RC_V,noise_RC_H 
