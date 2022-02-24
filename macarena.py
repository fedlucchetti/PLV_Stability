
import sys, os, glob, json, random, time, math
import pandas as pd
import numpy as np
from concurrent import futures
from scipy import signal
from scipy.signal import hilbert
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from functools import reduce

class PLV():
    name = "PLV"

    def __init__(self):
        self.N     = 25
        self.M     = 100
        self.epoch = 2048
        self.buff  = self.epoch*8
        self.fs    = 24414
        Nf         = int(self.epoch/2)
        dif        = self.fs/self.epoch
        self.f     = dif*np.array([x for x in range(Nf)])

        try:
            self.path2json = sys.argv[1]+' '+sys.argv[2]
        except:
            self.path2json = sys.argv[1]
        self.path      = os.path.dirname(self.path2json) + '/sAVG/'
        self.files     = glob.glob(self.path+'/*.xls*')
        if len(self.files)>438: self.files = self.files[-438:]
        self.r_im      = np.loadtxt('real_sol.txt')
        self.i_im      = np.loadtxt('imag_sol.txt')
        self.subchs    = ['VR','VR-noise','VC','VC-noise','HR','HR-noise','HC','HC-noise']
        self.chs       = ['Channel-V', 'Channel-H']
        self.SCstring  = ['EFR','EFR**','EFR***','F1','F2','CDT','CDT*','Noise']
        f              = lambda x: x.replace(',','.')
        self.conv      = {'VR':f,'VR-noise':f,'VC':f,'VC-noise':f,'HR':f,'HR-noise':f,'HC':f,'HC-noise':f}

        with open(self.path2json) as data_file: data = json.load(data_file)
        self.f1        = float(data["MetaData"]["Stimulus"]["F1"])
        self.f2        = float(data["MetaData"]["Stimulus"]["F2"])
        self.frequency = {'EFR':self.f2-self.f1,'EFR**':2*(self.f2-self.f1),'EFR***':3*(self.f2-self.f1),'F1':self.f1,'F2':self.f2,'CDT':2*self.f1-self.f2,'CDT*':2*self.f2-self.f1,'Noise':0.1}
        self.Noise     = {'Channel-V':data['FFR']["Channel-V"]['Noise']['AVG']['Waveform'],'Channel-H':data['FFR']["Channel-H"]['Noise']['AVG']['Waveform']}

        self.SCstring_detected  = {}
        self.ndlist = []
        for ch in self.chs:
            self.SCstring_detected[ch] = []
            for sc in self.SCstring[:-1]:
                fc             = self.frequency[sc]
                f_bin          = np.where(self.f>=fc)[0][0]
                signalSpectrum = np.abs(np.fft.fft(data['FFR'][ch][sc]['AVG']['Waveform']))
                noiseSpectrum  = np.abs(np.fft.fft(data['FFR'][ch]['Noise']['AVG']['Waveform']))
                mean_amplitude = (np.mean(signalSpectrum[f_bin-3:f_bin+3])-np.mean(noiseSpectrum[f_bin-3:f_bin+3]))/np.std(noiseSpectrum[f_bin-3:f_bin+3])
                try:
                    snr = 10*math.log10(np.mean(signalSpectrum[f_bin-3:f_bin+3])/np.mean(noiseSpectrum[f_bin-3:f_bin+3]))
                    data["FFR"][ch][sc]["Analysis"]['S/N'] = snr
                except:
                    pass

                #If the peak amplitude is 2 stardard deviations above the noise -> SC is detected -> Calculate PLV
                if np.abs(mean_amplitude)>2:
                    self.SCstring_detected[ch].append(sc)
                else:
                    self.ndlist.append(sc+ch)

            self.SCstring_detected[ch].append('Noise')

        #Delete PLV from all SCs
        for ch in self.chs:
            for sc in self.SCstring:
                data["FFR"][ch][sc]["Analysis"]['PLV'] = ""
        with open(self.path2json, 'w') as outfile: json.dump(data, outfile,ensure_ascii=False)

        #Create filter coefficients ONCE
        if self.frequency['EFR']>200:
            fmin = (self.frequency['EFR']-70)/(self.fs/2)
        elif self.frequency['EFR']<200 and self.frequency['EFR']>150:
            fmin = 100/(self.fs/2)
        elif self.frequency['EFR']<150:
            fmin = 80/(self.fs/2)
        fmax = (self.frequency['CDT']+1500)/(self.fs/2)
        self.b,self.a = signal.butter(4, [fmin,fmax], 'bandpass')


    def filterSC(self,_sc,sc):
        return signal.filtfilt(self.b,self.a,_sc)

    def subAVG(self,avg_path):
        try:
            df = pd.read_table(avg_path, header=0, names=self.subchs, usecols=self.subchs, skiprows=1, decimal=',',engine='c')
            _avg = df.to_numpy(dtype=np.float32)
        except: #There are some sAVG values with corrupted format such as: '-3.5600E-140E-6'
            df = pd.read_table(self.files[0], header=0, names=self.subchs, usecols=self.subchs, skiprows=1, converters=self.conv, engine='python')
            _avg = df.to_numpy(dtype=np.float32)
            print('rised Exception')

        return _avg

    def get_subAVG(self):
        _subAVG = []
        with futures.ThreadPoolExecutor() as executor:
            getsubAVG = {executor.submit(self.subAVG, _file): _file for _file in self.files}
            for future in futures.as_completed(getsubAVG):
                _subAVG.append(future.result())
        return _subAVG


    def subSC(self,sub_avg):
        R       = {'Channel-V': np.zeros((8,self.epoch)), 'Channel-H': np.zeros((8,self.epoch))}
        R_ht    = {'Channel-V': np.zeros((8,self.epoch)), 'Channel-H': np.zeros((8,self.epoch))}
        subRC   = {'Channel-V': {'R': np.split(sub_avg[:,0], 8), 'C': np.split(sub_avg[:,2], 8)},\
                   'Channel-H': {'R': np.split(sub_avg[:,4], 8), 'C': np.split(sub_avg[:,6], 8)}}
        sub_sc  = {}

        #Response
        for ch in self.chs:
            sub_sc[ch] = {}
            R[ch][0]    = np.sum(np.stack((subRC[ch]['R'][0],subRC[ch]['R'][4],subRC[ch]['R'][5],subRC[ch]['R'][6],subRC[ch]['R'][7]), axis=0), axis=0)/5
            R_ht[ch][0] = np.imag(hilbert(R[ch][0]))
            R[ch][4]    = np.sum(np.stack((subRC[ch]['C'][0],subRC[ch]['C'][4],subRC[ch]['C'][5],subRC[ch]['C'][6],subRC[ch]['C'][7]), axis=0), axis=0)/5
            R_ht[ch][4] = np.imag(hilbert(R[ch][4]))

            for i in range(1,4):
                R[ch][i]      = subRC[ch]['R'][i]
                R_ht[ch][i]   = np.imag(hilbert(R[ch][i]))
                R[ch][i+4]    = subRC[ch]['C'][i]
                R_ht[ch][i+4] = np.imag(hilbert(R[ch][i+4]))

            a = np.matmul(self.i_im,R_ht[ch])
            b = np.matmul(self.r_im,R[ch])

            for j in range(len(self.SCstring)-1):
                sc_string = self.SCstring[j]
                sub_sc[ch][sc_string] = (b[j]-a[j])/16
        #Noise
        sub_sc['Channel-V']['Noise'] = (np.sum(np.split(sub_avg[:,1], 8), axis=0) + np.sum(np.split(sub_avg[:,3], 8), axis=0))/16
        sub_sc['Channel-H']['Noise'] = (np.sum(np.split(sub_avg[:,5], 8), axis=0) + np.sum(np.split(sub_avg[:,7], 8), axis=0))/16

        return sub_sc

    def subPLV(self,_subAVGs):
        l_subchs = len(self.subchs)
        avg = np.zeros((self.buff, l_subchs))

        for i in range(l_subchs):
            aux_avg = np.zeros(self.buff)
            for s_avg in _subAVGs:
                try:
                    aux_avg += s_avg[:,i]
                except:
                    continue
            avg[:,i] = aux_avg/self.N

        _subSC = self.subSC(avg)
        polar_pd = {}
        for ch in self.chs:
            polar_pd[ch] = {}
            for sc in self.SCstring_detected[ch]:
                filt_sc             = self.filterSC(_subSC[ch][sc],sc)
                ip_wav              = np.unwrap(np.angle(hilbert(filt_sc)))
                if sc == "Noise":
                    y               = self.Noise[ch]
                else:
                    y               = np.sin(2 * np.pi * self.frequency[sc] * np.arange(self.epoch) / self.fs)
                ip_y                = np.unwrap(np.angle(signal.hilbert(y)))
                phase_diff          = np.abs(ip_wav - ip_y)
                _plv                = np.array([np.exp(complex(0,i)) for i in phase_diff],dtype='complex')
                polar_pd[ch][sc]    = np.abs(np.sum(_plv, axis=0)/len(_plv))

                if ch=='Channel-V' and sc=='EFR':
                    ip_V = ip_wav
                if ch=='Channel-H' and sc=='EFR':
                    ip_H = ip_wav

        if 'EFR' in self.SCstring_detected['Channel-V'] and 'EFR' in self.SCstring_detected['Channel-H']:
            phase_diff = np.mean(np.subtract(ip_V,ip_H))
        else:
            phase_diff = 0

        return polar_pd,phase_diff

    def get_subPLV(self,rand_subSCs):
        _subPLV = []

        with futures.ThreadPoolExecutor() as executor:
            getsubPLV = {executor.submit(self.subPLV,_sub): _sub for _sub in rand_subSCs}
            for future in futures.as_completed(getsubPLV):
                _subPLV.append(future.result())

        pd_dist = [str(sub[1]) for sub in _subPLV]

        with open(self.path2json) as json_file: data = json.load(json_file)
        data["FFR"]["Channel-V"]["EFR"]["Analysis"]['InstPhase'] = ",".join(pd_dist)

        for ch in self.chs:
            for sc in self.SCstring_detected[ch]:
                sub_plvs = np.zeros(self.M)
                for t, sub in enumerate(_subPLV):
                    sub_plvs[t] = sub[0][ch][sc]

                plv_dist = []
                for _plv in sub_plvs:
                    plv_dist.append(str(round(_plv,3)))
                data["FFR"][ch][sc]["Analysis"]['PLV'] = ",".join(plv_dist) # Save PLV in .json

        with open(self.path2json, 'w') as outfile: json.dump(data, outfile,ensure_ascii=False)

    def init_nd_ns(self): #non detected and non stationary
        with open(self.path2json) as json_file: data = json.load(json_file)
        for ch in self.chs:
            plv_noise = np.asarray(data["FFR"][ch]["Noise"]["Analysis"]["PLV"].split(","), dtype=np.float64, order='C')
            _,pv_noise = shapiro(plv_noise)

            for sc in ['EFR','EFR**','EFR***','F1','F2','CDT','CDT*']:
                if data["FFR"][ch][sc]["Analysis"]["PLV"] != '':
                    _plv=np.asarray(data["FFR"][ch][sc]["Analysis"]["PLV"].split(","), dtype=np.float64, order='C')
                    _, pv = shapiro(_plv)
                    if pv > 0.05 and pv_noise > 0.05:
                        if np.var(_plv) == np.var(plv_noise):
                            #ind t-test
                            _, pvalue = stats.ttest_ind(_plv,plv_noise,alternative='greater')
                        else:
                            #rel t-test
                            _, pvalue = stats.ttest_rel(_plv,plv_noise,alternative='greater')
                        if pvalue > 0.16: #1 std deviation #Detected but non-stationary
                           print("non-stationary:"+ch+" "+sc)
                        #   Initialization latencies to check in the analysis program
                        #    data["FFR"][ch][sc]["Analysis"]['Latency'] = 40.0
                        #    data["FFR"][ch][sc]["Analysis"]['Lenght'] = 83.0
                else: # Non-detected
                   print("non-detected:"+ch+" "+sc)
                # Initialization latencies to check in the analysis program
                #    data["FFR"][ch][sc]["Analysis"]['Latency'] = 20
                #    data["FFR"][ch][sc]["Analysis"]['Lenght'] = 83.0
            with open(self.path2json, 'w') as outfile: json.dump(data, outfile,ensure_ascii=False)
        return

if __name__ == '__main__':
    plv = PLV()
    M   = plv.M
    N   = plv.N

    start_time = time.time()
    if len(plv.files)>180:
        subAVGs = plv.get_subAVG()
        random_subAVGs = []
        for i in range(M):
            random_subAVGs.append(random.sample(subAVGs,N))
        plv.get_subPLV(random_subAVGs)
    plv.init_nd_ns()

    print("Tiempo: ",(time.time()-start_time))
