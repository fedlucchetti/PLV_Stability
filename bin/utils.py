import numpy as np
import sys,os,glob, csv
from os.path import join, split
from tqdm import tqdm
from bin import bcolors
bcolors=bcolors.bcolors()
N = 2048
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

    def load_all_trials(self,files):
        if len(files)==0:
            print(bcolors.FAIL,"No sAVG files found, EXIT")
            sys.exit()
        print("\n",bcolors.HEADER,"START:",bcolors.ENDC," Loading all trials into memory")
        RC8_V,RC8_H = np.zeros([len(files),N,16]),np.zeros([len(files),N,16])

        for file_index,file in enumerate(tqdm(files)):
            data                                = self.open_sAVG_file(file)
            RC8_V[file_index],RC8_H[file_index] = self.format_to_R_waveforms(data)
        print("\n",bcolors.OKGREEN,"DONE:",bcolors.ENDC,"  Loading all trials into memory \n",)
        return RC8_V,RC8_H

    def format_to_R_waveforms(self,data):
        waveforms_RC_V = np.zeros([N,16])
        waveforms_RC_H = np.zeros([N,16])
        for block_id in range(8):
            waveforms_RC_V[:,block_id]   = data[block_id*N:(block_id+1)*N,0]
            waveforms_RC_V[:,8+block_id] = data[block_id*N:(block_id+1)*N,2]
            waveforms_RC_H[:,block_id]   = data[block_id*N:(block_id+1)*N,4]
            waveforms_RC_H[:,8+block_id] = data[block_id*N:(block_id+1)*N,6]
        return waveforms_RC_V,waveforms_RC_H
