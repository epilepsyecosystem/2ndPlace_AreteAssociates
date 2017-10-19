import scipy.io as sio
import numpy as np
import glob
import os
import scipy.misc as misc
from sklearn import preprocessing
import scipy as sp
import scipy.signal as spsig
import pandas as pd
import scipy.stats as spstat
import json

#def get_data(file):
#	matfile = sio.loadmat(file)
#	data = (matfile['dataStruct']['data'][0,0]).T
#	return data
	
def get_data(file):
        try:
                matfile = sio.loadmat(file)
                data = (matfile['dataStruct']['data'][0,0]).T
                return data
        except Exception:
                print 'bad file:', file
                return np.zeros([16,400*10*60])

def long_features(pat, outfile, datapath):

        #pat = 3
        #outfile='pat_'+str(pat)+'_long_newtest_sub.csv'
        # file path for the new_test data
        #f = '/mnt/am02_scratch/blang/kaggle_data/test_'+str(pat)+'_new/*mat'
        # file path for the training and hold-out testing 
        #f = '/mnt/am02_scratch/blang/kaggle_data/CV/pat_'+str(pat)+'/train/*mat'
        f = datapath + '/*mat'
        
        pat_num = pat
        ff = glob.glob(f)
        
        label=[str(os.path.basename(n)) for n in ff]
        output = []
        featureList = []
        mydata = []
        bands=[0.1,4,8,12,30,70]
        for i in range(len(ff)):
                print float(i)/float(len(ff))
                output = []
                featureList = []
                if os.path.basename(ff[i]) == '1_45_1.mat':
                        continue
                data = get_data(ff[i])
                data = preprocessing.scale(data, axis=1,with_std=True)
                featureList.append('File')
                output.append(label[i])
                featureList.append('pat')
                output.append(pat_num)
        
                # get correlation Coef. this will be 16x16
                h=np.corrcoef(data)
                h=np.nan_to_num(h)
                # only want upper triangle
                ind = np.triu_indices(16, 1)
                htri = h[ind]
                for ii in range(np.size(htri)):
                        featureList.append('coef%i'%(ii))
                        output.append(htri[ii])

                c,v = np.linalg.eig(h)
                c.sort()
                c = np.real(c)
                for e in range(len(c)):
                        featureList.append('coef_timeEig%i'%(e))
                        output.append(c[e])

                for j in range(16):
                        hold = spsig.decimate(data[j,:],5,zero_phase=True)
                        featureList.append('sigma%i'%(j))
                        output.append(hold.std())
                        featureList.append('kurt%i'%(j))
                        output.append(spstat.kurtosis(hold))
                        featureList.append('skew%i'%(j))
                        output.append(spstat.skew(hold))
                        featureList.append('zero%i'%(j))
                        output.append(((hold[:-1] * hold[1:]) < 0).sum())
                        diff = np.diff(hold, n=1) 
                        diff2 = np.diff(hold,n=2)
                        featureList.append('sigmad1%i'%(j))
                        output.append(diff.std())
                        featureList.append('sigmad2%i'%(j))
                        output.append(diff2.std())
                        featureList.append('zerod%i'%(j))
                        output.append(((diff[:-1] * diff[1:]) < 0).sum())
                        featureList.append('zerod2%i'%(j))
                        output.append(((diff2[:-1] * diff2[1:]) < 0).sum())
                        featureList.append('RMS%i'%(j))
                        output.append(np.sqrt((hold**2).mean()))
                        f, psd = spsig.welch(hold, fs = 80)
                        psd[0] = 0
                        featureList.append('MaxF%i'%(j))
                        output.append(psd.argmax())
                        featureList.append('SumEnergy%i'%(j))
                        output.append(psd.sum())
                        psd /= psd.sum()
                        for c in range(1,len(bands)):
                                featureList.append('BandEnergy%i%i'%(j,c))
                                output.append(psd[(f>bands[c-1])&(f<bands[c])].sum())
                        featureList.append('entropy%i'%(j))
                        output.append(-1.0*np.sum(psd[f>bands[0]]*np.log10(psd[f>bands[0]])))
                        #pdb.exit()
                        featureList.append('Mobility%i'%(j))
                        output.append(np.std(diff)/hold.std())
                        featureList.append('Complexity%i'%(j))
                        output.append(np.std(diff2)*np.std(hold)/(np.std(diff)**2.))
                
                
                mydata.append(pd.DataFrame({'Features':output},index=featureList).T)
        trainSample = pd.concat(mydata,ignore_index=True)
        trainSample.to_csv(outfile)
        return 1

def short_features(pat, outfile, datapath):
        
        #pat = 3
        #outfile = 'pat_'+str(pat)+'_short_newtest_sub.csv'
        #file path for training and hold-out testing
        #f = '/mnt/am02_scratch/blang/kaggle_data/CV/pat_'+str(pat)+'/train/*mat'
        #file path for the new testing 
        #f = '/mnt/am02_scratch/blang/kaggle_data/test_'+str(pat)+'_new/*mat'
        f = datapath+'/*mat'
        ff = glob.glob(f)
        
        label=[str(os.path.basename(n)) for n in ff]
        output = []
        featureList = []
        mydata = []
        bands=[0.1,4,8,12,30,70,180]
        rate = 400.
        for i in range(len(ff)):
                print float(i)/float(len(ff))
                data_full = get_data(ff[i])
                output = []
                featureList = []
                featureList.append('File')
                output.append(label[i])
                featureList.append('pat')
                output.append(pat)
                
                for j in range(19):
                        if os.path.basename(ff[i]) == '1_45_1.mat':
                                continue
                        data = data_full[:,j*int(rate*60/2):(j)*int(rate*60/2) + int(rate)*60]
                        data = preprocessing.scale(data, axis=1,with_std=True)
                        
                        for k in range(16):
                                hold = data[k,:]
                                f,psd = spsig.welch(hold, fs=400, nperseg=2000)
                                psd = np.nan_to_num(psd)
                                psd /= psd.sum()
                                for c in range(1,len(bands)):
                                        featureList.append('BandEnergy_%i_%i_%i'%(j,k,c))
                                        output.append(psd[(f>bands[c-1])&(f<bands[c])].sum())
                mydata.append(pd.DataFrame({'Features':output},index=featureList).T)
        trainSample = pd.concat(mydata,ignore_index=True)
        trainSample.to_csv(outfile)
        return 1

def main():
        
        feat = json.load(open('SETTINGS.json'))
        keys = feat.keys()
        pat = feat['pat']
        if feat['make_test'] == 1:
                outfile = feat['feat']+'/pat_'+str(pat)+'_long_newtest_sub.csv'
                l = long_features(pat,outfile,feat['test'])
                outfile = feat['feat']+'/pat_'+str(pat)+'_short_newtest_sub.csv'
                s = short_features(pat, outfile, feat['test'])
        if feat['make_train'] == 1:
                outfile = feat['feat']+'/pat_'+str(pat)+'_long_train.csv'
                l = long_features(pat,outfile,feat['train'])
                outfile = feat['feat'] + '/pat_'+str(pat)+'_short_train.csv'
                s = short_features(pat, outfile, feat['train'])
        if feat['make_hold'] == 1:
                outfile = feat['feat']+'/pat_'+str(pat)+'_long_test.csv'
                l = long_features(pat,outfile,feat['hold-out'])
                outfile = feat['feat'] + '/pat_'+str(pat)+'_short_test.csv'
                s = short_features(pat, outfile, feat['hold-out'])

if __name__ == "__main__":
        main()
