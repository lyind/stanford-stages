# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:58:10 2017

@author: jens
@modifer: hyatt
@modifier: neergaard
# from: inf_eval --> to: inf_generate_hypnodensity
"""
import itertools  # for extracting feature combinations
import os  # for opening os files for pickle.
import pickle
from pathlib import Path

# for profiling
from tempfile import gettempdir

import skimage_extract as skimage
import numpy as np
import pyedflib
import pywt  # wavelet entropy
import scipy.io as sio  # for noise level
import scipy.signal as signal  # for edf channel sampling and filtering
import tensorflow as tf
from scipy.fftpack import fft, ifft, irfft, fftshift

from inf_config import ACConfig
from inf_network import SCModel
from inf_tools import myprint
import multiprocessing
import threading
import concurrent.futures
import re
import pdb
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x)
    div = np.repeat(np.expand_dims(np.sum(e_x, axis=1), 1), 5, axis=1)
    return np.divide(e_x, div)

def softmax(x):
    # e_x = np.exp(x - np.max(x))
    # return e_x / e_x.sum()
    e_x = np.exp(x)
    div = np.repeat(np.expand_dims(np.sum(e_x, axis=1), 1), 5, axis=1)
    return np.divide(e_x, div)

def softmax(x):
    e_x = np.exp(x)
    div = np.repeat(np.expand_dims(np.sum(e_x, axis=1), 1), 5, axis=1)
    return np.divide(e_x, div)

class Hypnodensity(object):

    ORDER = ['C3','C4','O1','O2', 'EOG-L', 'EOG-R', 'EMG']

    run_count = 0

    def __init__(self, appConfig):
        self.config = appConfig
        self.hypnodensity = list()
        self.Features = HypnodensityFeatures(appConfig)
        self.CCsize = appConfig.CCsize

        self.channels = appConfig.channels
#        self.loaded_channels = appConfig.loaded_channels
        self.edf_pathname = appConfig.edf_path
        self.encodedD = []
        self.fs = int(appConfig.fs)
        self.fsH = appConfig.fsH
        self.fsL = appConfig.fsL
        self.lightsOff = appConfig.lightsOff
        self.lightsOn = appConfig.lightsOn
        self.cpu_max = appConfig.cpu_max

        self.cached_hypnodensity = None


    def evaluate(self):
        p = Path(self.edf_pathname)
        p = Path(p.with_suffix('.pkl'))

        h = Path(self.edf_pathname)
        h = Path(h.with_suffix('.hypno_pkl'))

        if (p.exists()):

            myprint('Loading previously saved encoded data')
            with p.open('rb') as fp:
                self.encodedD = pickle.load(fp)
        else:
            myprint('Load EDF')
            self.loadEDF()

            myprint('Load noise level')
            self.psg_noise_level()

            self.filtering()

            print('filtering done')

            print('Encode')
            self.encoding()

            # pickle our file
            with p.open('wb') as fp:
                pickle.dump(self.encodedD, fp)
                myprint("pickling done")

        if (h.exists()):
            myprint('Loading previously saved hypnodensity')
            with h.open('rb') as fp:
                self.hypnodensity = pickle.load(fp)
        else:
            myprint('Score data')
            self.score_data()
            # pickle our file
            with h.open('wb') as fp:
                pickle.dump(self.hypnodensity, fp)
                myprint("Hypnodensity pickled")

    # compacts hypnodensity, possibly from mutliple models, into one Mx5 probability matrix.
    def get_hypnodensity(self):
        if self.cached_hypnodensity is None:
            av = np.zeros(self.hypnodensity[0].shape)  # for example, 2144, 5)

            for i in range(len(self.hypnodensity)):
                av += self.hypnodensity[i]

            av = np.divide(av, len(self.hypnodensity))
            self.cached_hypnodensity = av

        return self.cached_hypnodensity

    # 0 is wake, 1 is stage-1, 2 is stage-2, 3 is stage 3/4, 5 is REM
    def get_hypnogram(self):
        hypno = self.get_hypnodensity()
        hypnogram = np.argmax(hypno, axis=1) # 0 is wake, 1 is stage-1, 2 is stage-2, 3 is stage 3/4, 4 is REM
        hypnogram[hypnogram==4]=5     # Change 4 to 5 to keep with the conventional REM indicator
        return hypnogram


    def get_features(self, modelName, idx):
        selected_features = self.config.narco_prediction_selected_features
        X = self.Features.extract(self.hypnodensity[idx])
        X = self.Features.scale_features(X, modelName)
        return X[selected_features].T

    # Use 5 minute sliding window.
    def extract_hjorth(self, x, dim=5 * 60, slide=5 * 60):

        # Length of first dimension
        dim = dim * self.fs

        # Overlap of segments in samples
        slide = slide * self.fs

        # Creates 2D array of overlapping segments
        D = skimage.view_as_windows(x, dim, dim).T

        # Extract Hjorth params for each segment
        dD = np.diff(D, 1, axis=0)
        ddD = np.diff(dD, 1, axis=0)
        mD2 = np.mean(D ** 2, axis=0)
        mdD2 = np.mean(dD ** 2, axis=0)
        mddD2 = np.mean(ddD ** 2, axis=0)

        top = np.sqrt(np.divide(mddD2, mdD2))

        mobility = np.sqrt(np.divide(mdD2, mD2))
        activity = mD2
        complexity = np.divide(top, mobility)

        hjorth = np.array([activity, complexity, mobility])
        hjorth = np.log(hjorth + np.finfo(float).eps)
        return hjorth

    def mob(self, B):
        diff = np.diff(B, axis=0)
        var = np.var(diff, axis=0)

        return np.sqrt(np.divide(var, np.var(B, axis=0)))

    def encoding(self):

        def encode_data(x1, x2, dim, slide, fs):

            # Length of the first dimension and overlap of segments
            dim = int(fs * dim)
            slide = int(fs * slide)

            # Create 2D array of overlapping segments
            zero_vec = np.zeros(dim // 2)
            input2 = np.concatenate((zero_vec, x2, zero_vec))
            D1 = skimage.view_as_windows(x1, dim, slide).T
            D2 = skimage.view_as_windows(input2, dim * 2, slide).T
            zero_mat = np.zeros((dim // 2, D1.shape[1]))
            D1 = np.concatenate([zero_mat, D1, zero_mat])

            keep_dims = D1.shape[1]
            D2 = D2[:, :keep_dims]
            D1 = D1[:, :keep_dims]

            # Fast implementation of auto/cross-correlation
            C = fftshift(
                np.real(ifft(fft(D1, dim * 2 - 1, axis=0) * np.conj(fft(D2, dim * 2 - 1, axis=0)), axis=0)),
                axes=0).astype(dtype=np.float32)

            # Remove mirrored part
            C = C[dim // 2 - 1: - dim // 2]


            #folding up the downside and adding together (halves data size)
#            C = np.split(C,2, axis=0)
#            C[1] = np.flip(C[1], axis=0)
#            C = np.sum(C, axis=0)

            #reducing resolution along y-axis (halves data size)
#            C = np.multiply(np.sum([C[0:C.shape[0]:2],C[1::2]], axis=0),0.5)
            
            #adding a flipped version on the downside again(->mirrors again, upside and downside identical now); for checking for quality
#            Cm = np.flip(C, axis=0)
#            C = np.concatenate((C,Cm), axis=0)
            
            #scaling after reshape/ Scale data with log modulus
            scale = np.log(np.max(np.abs(C) + 1, axis=0) / dim)
            C = C[..., :] / (np.amax(np.abs(C), axis=0) / scale)

            return C

        count = -1
        enc = []

#            if c in self.loaded_channels:
#                # append autocorrelations
#                enc.append(encode_data(self.loaded_channels[c], self.loaded_channels[c], self.CCsize[c], 0.25, self.fs))
 
        info = {}
        for ch in self.loaded_channels:
            info[ch] = [self.loaded_channels[ch], self.CCsize[ch], 0.25, self.fs]

        print('number of CPUs: ' + str(multiprocessing.cpu_count()))
        cpu_set = min(self.cpu_max ,max(1, (multiprocessing.cpu_count()-1)))
        
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_set) as executor:
            for ch in self.ORDER:
                if ch in info:
                    ch_info = info[ch]
                    future = executor.submit(encode_data, ch_info[0], ch_info[0], ch_info[1], ch_info[2], ch_info[3])
                    futures.append(future)
        
        for future in futures:
            enc.append(future.result())

        # Append eog cross correlation
        enc.append(encode_data(self.loaded_channels['EOG-L'], self.loaded_channels['EOG-R'], self.CCsize['EOG-L'], 0.25, self.fs))
        min_length = np.min([x.shape[1] for x in enc])
        enc = [v[:, :min_length] for v in enc]
        # Central, Occipital, EOG-L, EOG-R, EOG-L/R, chin
 #For visualizing the Data after CC; names = ['Cx','Ox','EOG-L','EOG-R','EOG-cross','EMG']
 #       plt.imshow(enc[0][:,41000:41120])
 #       plt.savefig('Cx')
 #       plt.imshow(enc[1][:,41000:41120])
 #       plt.savefig('Ox')
 #       plt.imshow(enc[4][:,41000:41120])
 #       plt.savefig('EMG')
 #       plt.imshow(enc[5][:,41000:41120])
 #       plt.savefig('EOG-cross')
 #       plt.imshow(enc[2][:,41000:41120])
 #       plt.savefig('EOG-L')
 #       plt.imshow(enc[3][:,41000:41120])
 #       plt.savefig('EOG-R')
        
        enc = np.concatenate([enc[0], enc[1], enc[2], enc[3], enc[5], enc[4]], axis=0)
        self.encodedD = enc

        # Needs double checking as magic numbers are problematic here and will vary based on configuration settings.  @hyatt 11/12/2018
        # Currently, this is not supported as an input json parameter, but will need to adjust accordingly if this changes.
        # Note: This extracts after the lightsOff epoch and before lightsOn epoch as python is 0-based.
        if isinstance(self.lightsOff, int):
            self.encodedD = self.encodedD[:,
                            4 * 30 * self.lightsOff:4 * 30 * self.lightsOn]


    def loadEDF(self):
        self.loaded_channels = {}

        loadSignal_lock = threading.Lock()
        def load_channel(target_label, original_label, source_edf, edf_labels):
            i = edf_labels.index(original_label)
            ch_signal = None
            with loadSignal_lock:
                ch_signal = source_edf.readSignal(i)
            dimension = source_edf.getPhysicalDimension(i).lower()
            if dimension == 'mv':
                ch_signal *= 1e3
            elif dimension == 'v':
                ch_signal *= 1e6
            original_fs = int(source_edf.samplefrequency(i))
            print(target_label + ': original samplerate: ' + str(original_fs));
            print(target_label + ': resampling: ' + str(self.fs));
            resampled_ch = self.resample(ch_signal, original_fs, self.fs)
            print(target_label + ': Resampling done')
            return resampled_ch

        def load_and_reference_channel(channel_state, target_label):
            label = channel_state['label']
            res_ch = load_channel(target_label, label, edf, Labels)
            ref = channel_state['ref_label']
            if ref and not channel_state['isReferenced']:
                if ref in self.loaded_channels:
                    res_ref = self.loaded_channels[ref]
                else:
                    res_ref = self.loaded_channels[ref] = load_channel(ref, ref, edf, Labels)
                
                print(target_label + ': referencing ' + label + ' with ' + ref)
                self.loaded_channels[target_label] = np.subtract(res_ch, res_ref)
            else:
                self.loaded_channels[target_label] = res_ch
        
        with pyedflib.EdfReader(str(self.edf_pathname)) as edf:
            Labels = edf.getSignalLabels()
            Channels = { 'O1': {'expr': ".*O1.*", 'ref': ['A2', 'M2'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'O2': {'expr': "(^.{0}|[^pPaA])O2.*", 'ref': ['A1', 'M1'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'C3': {'expr': ".*C3.*", 'ref': ['A2', 'M2'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'C4': {'expr': ".*C4.*", 'ref': ['A1', 'M1'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'EOG-L': {'expr': "EOG.?([1Ll]|[eE][1lL])", 'ref': ['A1', 'A2', 'M1', 'M2'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'EOG-R': {'expr': "EOG.?([2Rr]|[eE][2rR])", 'ref': ['A1', 'A2', 'M1', 'M2'], 'label': None, 'isReferenced': False, 'ref_label': None }
                    }
                    
            def find_reference_label(ref_candidates):
                for ref_label in Labels:
                    for r in ref_candidates:
                        if r in ref_label:
                            return ref_label
                return None
                
            def label_is_reference(ref_candidates, label):
                for r in ref_candidates:
                    if r in label:
                        return True
                return False

#finding channels and check if referenced
            for label in Labels:
                for ch in Channels:
                    channel_state = Channels[ch]
                    ident = re.search(channel_state['expr'], label)
                    if ident and not channel_state['isReferenced']:
                        print(ch + ': identified ' + label + ' as: ' + ch)
                        channel_state['label'] = label
                        
                        channel_state['isReferenced'] = label_is_reference(channel_state['ref'], label)
                        if channel_state['isReferenced']:
                            print(ch + ': is already referenced')
                        else:
                            print(ch + ': appears unreferenced')
                            channel_state['ref_label'] = find_reference_label(channel_state['ref'])
                            if channel_state['ref_label']:
                                print(ch + ': identified ' + channel_state['ref_label'] + ' as suitable reference')
                            else:
                                print(ch + ': No reference found')
    
                emg_ident = re.search(".*[cC][hH][iI][nN].*", label)
                if emg_ident:
                    print('EMG: found chin-EMG as: ' + label)
                    Channels['EMG'] = {'expr': "some string to prevent an ERROR", 'label':label, 'ref_label': None}
            
            if 'EMG' not in Channels:
                for label in Labels:
                    ident = re.search(".*[eE][mM][gG].*", label)
                    if ident:
                        print('EMG: found chin-EMG as: ' + label)
                        Channels['EMG'] = {'label':label, 'ref_label': None}
                        break
            
#extracting data from edf, resampling and auto-referencing
            cpu_set = min(self.cpu_max ,max(1, (multiprocessing.cpu_count()-1)))
            
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_set) as executor:
                for ch in Channels:
                    channel_state = Channels[ch]
                    if channel_state['label']:
                        future = executor.submit(load_and_reference_channel, channel_state, ch)
                        futures.append(future)
            
            for future in futures:
                future.result()
            
#take out unnecessary channels
            Takeout = []
            for entry in self.loaded_channels:
                if entry not in self.ORDER:
                    Takeout.append(entry)
            for x in Takeout:
                del self.loaded_channels[x]

            retitle = 0
            if 'O1' not in self.loaded_channels and 'O2' not in self.loaded_channels:
                if 'C3' in self.loaded_channels and 'C4' in self.loaded_channels:
                    self.loaded_channels['O1'] = self.loaded_channels['C4']
                    retitle = retitle +1
            if retitle > 0:
                del self.loaded_channels['C4']


    def trim(self, ch):
        # 30 represents the epoch length most often used in standard hypnogram scoring.
        rem = len(self.loaded_channels[ch]) % int(self.fs * 30)
        # Otherwise, if rem == 0, the following results in an empty array
        if rem>0:
            self.loaded_channels[ch] = self.loaded_channels[ch][:-rem]

    def filtering(self):
        myprint('Filtering remaining signals')
        fs = self.fs

        Fh = signal.butter(5, self.fsH / (fs / 2), btype='highpass', output='ba')
        Fl = signal.butter(5, self.fsL / (fs / 2), btype='lowpass', output='ba')

        for ch in self.loaded_channels:
                myprint('Filtering {}'.format(ch))
                self.loaded_channels[ch] = signal.filtfilt(Fh[0], Fh[1], self.loaded_channels[ch])

                if fs > (2 * self.fsL):
                    self.loaded_channels[ch] = signal.filtfilt(Fl[0], Fl[1], self.loaded_channels[ch]).astype(
                        dtype=np.float32)

    def resample(self, data, original_frequency, target_frequency ):
        if original_frequency==500 or original_frequency==200:
            numerator = [[-0.0175636017706537, -0.0208207236911009, -0.0186368912579407, 0.0, 0.0376532652007562,
                0.0894912177899215, 0.143586518157187, 0.184663795586300, 0.200000000000000, 0.184663795586300,
                0.143586518157187, 0.0894912177899215, 0.0376532652007562, 0.0, -0.0186368912579407,
                -0.0208207236911009, -0.0175636017706537],
                [-0.050624178425469, 0.0, 0.295059334702992, 0.500000000000000, 0.295059334702992, 0.0,
                -0.050624178425469]]  # from matlab
            if original_frequency==500:
                s = signal.dlti(numerator[0], [1], dt=1. / target_frequency)
                resampled_ch = signal.decimate(data, original_frequency // target_frequency, ftype=s, zero_phase=False)
            elif original_frequency==200:
                s = signal.dlti(numerator[1], [1], dt=1. / target_frequency)
                resampled_ch = signal.decimate(data, original_frequency // target_frequency, ftype=s, zero_phase=False)
        else:
            resampled_ch = signal.resample_poly(data, target_frequency, original_frequency, axis=0, window=('kaiser', 5.0))
        return resampled_ch

    def psg_noise_level(self):

        # Only need to check noise levels when we have two central or occipital channels
        # which we should then compare for quality and take the best one.  We can test this
        # by first checking if there is a channel category 'C4' or 'O2'
        hasC4 = self.loaded_channels.get('C4')
        hasC3 = self.loaded_channels.get('C3')
        hasO2 = self.loaded_channels.get('O2')
        hasO1 = self.loaded_channels.get('O1')
        if all(['C4' in self.loaded_channels, 'C3' in self.loaded_channels]):
            noiseM = sio.loadmat(self.config.psg_noise_file_pathname, squeeze_me=True)['noiseM']
            meanV = noiseM['meanV'].item()  # 0 for Central,    idx_central = 0
            covM = noiseM['covM'].item()    # 1 for Occipital,  idx_occipital = 1

            #if hasC4:
            centrals_idx = 0
            unused_ch = self.get_loudest_channel(['C3','C4'],meanV[centrals_idx], covM[centrals_idx])
            del self.loaded_channels[unused_ch]



        if all(['O2' in self.loaded_channels, 'O1' in self.loaded_channels]):
            noiseM = sio.loadmat(self.config.psg_noise_file_pathname, squeeze_me=True)['noiseM']
            meanV = noiseM['meanV'].item()  # 0 for Central,    idx_central = 0
            covM = noiseM['covM'].item()    # 1 for Occipital,  idx_occipital = 1
            occipitals_idx = 1
            unused_ch = self.get_loudest_channel(['O1','O2'],meanV[occipitals_idx], covM[occipitals_idx])
            del self.loaded_channels[unused_ch]


    def get_loudest_channel(self, channelTags, meanV, covM):
        noise = np.zeros(len(channelTags))
        for [idx,ch] in enumerate(channelTags):
            noise[idx] = self.channel_noise_level(ch, meanV, covM)
        return channelTags[np.argmax(noise)]

        # for ch in channelTags:
        #     noise = self.channel_noise_level(ch, meanV, covM)
        #     if noise >= loudest_noise:
        #         loudest_noise = noise
        #         loudest_ch = ch
        # return loudest_ch

    def channel_noise_level(self, channelTag, meanV, covM):

        hjorth= self.extract_hjorth(self.loaded_channels[channelTag])
        noise_vec = np.zeros(hjorth.shape[1])
        for k in range(len(noise_vec)):
            M = hjorth[:, k][:, np.newaxis]
            x = M - meanV[:, np.newaxis]
            sigma = np.linalg.inv(covM)
            noise_vec[k] = np.sqrt(np.dot(np.dot(np.transpose(x), sigma), x))
            return np.mean(noise_vec)

    def run_data(dat, model, root_model_path, segsize):

        ac_config = ACConfig(model_name=model, is_training=False, root_model_dir=root_model_path, segsize_override=segsize)
        hyp = Hypnodensity.run(dat, ac_config)
        return hyp

    def get_acconfig(self):
        return self.ac_config

    def score_data(self):
        from pathlib import Path
        model = Path(self.config.hypnodensity_model_root_path)
        print("AC config hypnodensity path: " + str(model) + ', is_dir=' + str(model.is_dir()))
        
        self.hypnodensity = list()
        for l in self.config.models_used:
            hyp = Hypnodensity.run_data(self.encodedD, l, self.config.hypnodensity_model_root_path, self.config.segsize)
            hyp = softmax(hyp)
            self.hypnodensity.append(hyp)

    def segment(dat, ac_config):

        once = min(ac_config.eval_nseg_atonce, int((dat.shape[1]/ac_config.segsize)-1))
        
        # Get integer value for segment size using //
        n_seg = dat.shape[1] // ac_config.segsize

        dat = np.expand_dims(dat[:, :n_seg * ac_config.segsize], 0)

        num_batches = np.int(
            np.ceil(np.divide(dat.shape[2], (once * ac_config.segsize), dtype='float')))

        Nextra = np.int(np.ceil(num_batches * once * ac_config.segsize) % dat.shape[2])
        # why not:    Nextra = num_batches * ac_config.eval_nseg_atonce * ac_config.segsize - dat.shape[2]

        # fill remaining (nExtra) values with the mean value of each column
        meanF = np.mean(np.mean(dat, 2), 0) * np.ones([1, Nextra, dat.shape[1]])




        dat = np.transpose(dat, [0, 2, 1])
        dat = np.concatenate([dat, meanF], 1)

        prediction = np.zeros([num_batches * once, 5])

        return dat, Nextra, prediction, num_batches

    def run(dat, ac_config):

        ++Hypnodensity.run_count

        once = min(ac_config.eval_nseg_atonce, int((dat.shape[1]/ac_config.segsize)-1))

        thread_count = max(multiprocessing.cpu_count() - 1, 1)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
        
        with tf.Graph().as_default() as g:
            m = SCModel(ac_config)

            s = tf.train.Saver(tf.global_variables())

            print("AC config hypnodensity path",ac_config.hypnodensity_model_dir)

            builder = tf.profiler.ProfileOptionBuilder
            opts = builder(builder.time_and_memory()).order_by('micros').build()
            profile_dir = gettempdir() + 'mlstages_profile_' + str(Hypnodensity.run_count)

            with tf.contrib.tfprof.ProfileContext(profile_dir) as pctx:

                config = tf.ConfigProto(log_device_placement=False,
                    intra_op_parallelism_threads=thread_count,
                    inter_op_parallelism_threads=thread_count)

                with tf.Session(config=config) as session:

                    pctx.trace_next_step()
                    pctx.dump_next_step()
                    ckpt = tf.train.get_checkpoint_state(ac_config.hypnodensity_model_dir)
                    pctx.profiler.profile_operations(options=opts)

                    s.restore(session, ckpt.model_checkpoint_path)

                    state = np.zeros([1, ac_config.num_hidden * 2])

                    dat, Nextra, prediction, num_batches = Hypnodensity.segment(dat, ac_config)
                    for i in range(num_batches):
                        x = dat[:, i * once * ac_config.segsize:(i + 1) * once * ac_config.segsize,:]

                        pctx.trace_next_step()
                        pctx.dump_next_step()
                        est, _ = session.run([m.logits, m.final_state], feed_dict={
                            m.features: x,
                            m.targets: np.ones([once * ac_config.segsize, 5]),
                            m.mask: np.ones(once * ac_config.segsize),
                            m.batch_size: np.ones([1]),
                            m.initial_state: state
                        },
                        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
                        pctx.profiler.profile_operations(options=opts)

                        prediction[i * once:(i + 1) * once, :] = est


                    prediction = prediction[:-int(Nextra / ac_config.segsize), :]

                    return prediction


class HypnodensityFeatures(object):  # <-- extract_features

    def __init__(self, appConfig):
        self.config = appConfig
        self.meanV = []
        self.scaleV = []
        self.selected = []  # [1, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 299, 390, 405, 450, 467, 468, 470, 474, 476, 477]
        self.scale_path = appConfig.hypnodensity_scale_path  # 'scaling'
        # self.select_features_path = appConfig.hypnodensity_select_features_path
        # self.select_features_pickle_name = appConfig.hypnodensity_select_features_pickle_name  # 'narcoFeatureSelect.p'

    def extract(self, hyp):
        eps = 1e-10
        features = np.zeros([24 + 31 * 15])
        j = -1
        f = 10

        for i in range(5):
            for comb in itertools.combinations([0, 1, 2, 3, 4], i + 1):
                j += 1
                dat = np.prod(hyp[:, comb], axis=1) ** (1 / float(len(comb)))

                features[j * 15] = np.log(np.mean(dat) + eps)
                features[j * 15 + 1] = -np.log(1 - np.max(dat))

                moving_av = np.convolve(dat, np.ones(10), mode='valid')
                features[j * 15 + 2] = np.mean(np.abs(np.diff(moving_av)))

                features[j * 15 + 3] = self.wavelet_entropy(dat)  # Shannon entropy - check if it is used as a feature

                rate = np.cumsum(dat) / np.sum(dat)
                I1 = (i for i, v in enumerate(rate) if v > 0.05).__next__()
                features[j * 15 + 4] = np.log(I1 * 2 + eps)
                I2 = (i for i, v in enumerate(rate) if v > 0.1).__next__()
                features[j * 15 + 5] = np.log(I2 * 2 + eps)
                I3 = (i for i, v in enumerate(rate) if v > 0.3).__next__()
                features[j * 15 + 6] = np.log(I3 * 2 + eps)
                I4 = (i for i, v in enumerate(rate) if v > 0.5).__next__()
                features[j * 15 + 7] = np.log(I4 * 2 + eps)

                features[j * 15 + 8] = np.sqrt(np.max(dat) * np.mean(dat) + eps)
                features[j * 15 + 9] = np.mean(np.abs(np.diff(dat)) * np.mean(dat) + eps)
                features[j * 15 + 10] = np.log(self.wavelet_entropy(dat) * np.mean(dat) + eps)
                features[j * 15 + 11] = np.sqrt(I1 * 2 * np.mean(dat))
                features[j * 15 + 12] = np.sqrt(I2 * 2 * np.mean(dat))
                features[j * 15 + 13] = np.sqrt(I3 * 2 * np.mean(dat))
                features[j * 15 + 14] = np.sqrt(I4 * 2 * np.mean(dat))

        rem = (hyp.shape[0] % 2)
        if rem == 1:
            data = hyp[:-rem, :]
        else:
            data = hyp

        data = data.reshape([-1, 2, 5])
        data = np.squeeze(np.mean(data, axis=1))

        S = np.argmax(data, axis=1)

        SL = [i for i, v in enumerate(S) if v != 0]
        if len(SL) == 0:
            SL = len(data)
        else:
            SL = SL[0]

        RL = [i for i, v in enumerate(S) if v == 4]
        if len(RL) == 0:
            RL = len(data)
        else:
            RL = RL[0]

        # Nightly SOREMP

        wCount = 0;
        rCount = 0;
        rCountR = 0;
        soremC = 0;
        for i in range(SL, len(S)):
            if (S[i] == 0) | (S[i] == 1):
                wCount += 1
            elif (S[i] == 4) & (wCount > 4):
                rCount += 1
                rCountR += 1
            elif rCount > 1:
                soremC += 1
            else:
                wCount = 0
                rCount = 0

        # NREM Fragmentation
        nCount = 0
        nFrag = 0
        for i in range(SL, len(S)):
            if (S[i] == 2) | (S[i] == 3):
                nCount += 1
            elif ((S[i] == 0) | (S[i] == 1)) & (nCount > 3):
                nFrag += 1
                nCount = 0

        # W/N1 Bouts
        wCount = 0
        wBout = 0
        wCum = 0
        sCount = 0
        for i in range(SL, len(S)):
            if S[i] != 1:
                sCount += 1

            if (sCount > 5) & ((S[i] == 0) | (S[i] == 1)):
                wCount = wCount + 1
                if wCount < 30:
                    wCum = wCum + 1

            elif wCount > 4:
                wCount = 0
                wBout = wBout + 1


        features[-24] = self.logmodulus(SL * f)
        features[-23] = self.logmodulus(RL - SL * f)

        features[-22] = np.sqrt(rCountR)
        features[-21] = np.sqrt(soremC)
        features[-20] = np.sqrt(nFrag)
        features[-19] = np.sqrt(wCum)
        features[-18] = np.sqrt(wBout)

        ## Find out what features are used:...!
        features[-17:] = self.logmodulus(self.transitionFeatures(data))

        return features

    def select_features(self, threshold=1):

        if len(self.selected) == 0:
            try:
                with open(self.select_features_path / self.select_features_pickle_name, 'rb') as sel:
                    S = pickle.load(sel)
                    self.selected = S > threshold
            except FileNotFoundError as e:
                print("File not found ", e)

        return self.selected

    def logmodulus(self, x):
        return np.sign(x) * np.log(abs(x) + 1)

    def scale_features(self, features, sc_mod='unknown'):
        scaled_features = features
        if len(scaled_features.shape) == 1:
            scaled_features = np.expand_dims(scaled_features, axis=1)

        if len(self.meanV) == 0:
            try:
                with open(self.scale_path / (sc_mod + '_scale.p'), 'rb') as sca:
                    scaled = pickle.load(sca)
                self.meanV = np.expand_dims(scaled['meanV'], axis=1)[:, :, 0]
                self.scaleV = np.expand_dims(scaled['scaleV'], axis=1)[:, :, 0]
            except FileNotFoundError as e:
                print("File not found ", e)
                print("meanV set to 0 and scaleV set to 1")
                self.meanV = 0;
                self.scaleV = 1;

        scaled_features -= self.meanV
        scaled_features = np.divide(scaled_features, self.scaleV)

        scaled_features[scaled_features > 10] = 10
        scaled_features[scaled_features < -10] = -10

        return scaled_features

    def transitionFeatures(self, data):
        S = np.zeros(data.shape)
        for i in range(5):
            S[:, i] = np.convolve(data[:, i], np.ones(9), mode='same')

        S = softmax(S)

        cumR = np.zeros(S.shape)
        Th = 0.2;
        peakTh = 10;
        for j in range(5):
            for i in range(len(S)):
                if S[i - 1, j] > Th:
                    cumR[i, j] = cumR[i - 1, j] + S[i - 1, j]

            cumR[cumR[:, j] < peakTh, j] = 0

        for i in range(5):
            d = cumR[:, i]
            indP = self.find_peaks(cumR[:, i])
            typeP = np.ones(len(indP)) * i
            if i == 0:
                peaks = np.concatenate([np.expand_dims(indP, axis=1), np.expand_dims(typeP, axis=1)], axis=1)
            else:
                peaks = np.concatenate([peaks, np.concatenate([np.expand_dims(indP, axis=1),
                                                               np.expand_dims(typeP, axis=1)], axis=1)], axis=0)

        I = [i[0] for i in sorted(enumerate(peaks[:, 0]), key=lambda x: x[1])]
        peaks = peaks[I, :]

        remList = np.zeros(len(peaks))

        peaks[peaks[:, 1] == 0, 1] = 1
        peaks[:, 1] = peaks[:, 1] - 1

        if peaks.shape[0] < 2:
            features = np.zeros(17)
            return features

        for i in range(peaks.shape[0] - 1):
            if peaks[i, 1] == peaks[i + 1, 1]:
                peaks[i + 1, 0] += peaks[i, 0]
                remList[i] = 1
        remList = remList == 0
        peaks = peaks[remList, :]
        transitions = np.zeros([4, 4])

        for i in range(peaks.shape[0] - 1):
            transitions[int(peaks[i, 1]), int(peaks[i + 1, 1])] = np.sqrt(peaks[i, 0] * peaks[i + 1, 0])
        di = np.diag_indices(4)
        transitions[di] = None

        transitions = transitions.reshape(-1)
        transitions = transitions[np.invert(np.isnan(transitions))]
        nPeaks = np.zeros(5)
        for i in range(4):
            nPeaks[i] = np.sum(peaks[:, 1] == i)

        nPeaks[-1] = peaks.shape[0]

        features = np.concatenate([transitions, nPeaks], axis=0)
        return features

    def find_peaks(self, x):
        peaks = []
        for i in range(1, len(x)):
            if x[i - 1] > x[i]:
                peaks.append(i - 1)

        return np.asarray(peaks)

    def wavelet_entropy(self, dat):
        coef, freqs = pywt.cwt(dat, np.arange(1, 60), 'gaus1')
        Eai = np.sum(np.square(np.abs(coef)), axis=1)
        pai = Eai / np.sum(Eai)

        WE = -np.sum(np.log(pai) * pai)

        return WE
