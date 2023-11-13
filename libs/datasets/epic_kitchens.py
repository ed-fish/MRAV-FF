import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import torchaudio
import soundfile as sf
import math

from scipy import signal

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("epic")
class EpicKitchensDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        additional_feat_folder=None,
        
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        
        self.norm_mean =  -4.984795570373535
        self.norm_std =  3.7079780101776123

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        # "empty" noun categories on epic-kitchens
        assert len(label_dict) <= num_classes
        self.data_list = dict_db
        self.label_dict = label_dict
        
        self.use_addtional_feats = additional_feat_folder is not None
        self.additional_feat_folder = additional_feat_folder

        # dataset specific attributes
        empty_label_ids = self.find_empty_cls(label_dict, num_classes)
        self.db_attributes = {
            'dataset_name': 'epic-kitchens-100',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': empty_label_ids
        }

    def find_empty_cls(self, label_dict, num_classes):
        # find categories with out a data sample
        if len(label_dict) == num_classes:
            return []
        empty_label_ids = []
        label_ids = [v for _, v in label_dict.items()]
        for id in range(num_classes):
            if id not in label_ids:
                empty_label_ids.append(id)
        return empty_label_ids
    
    # def _wav2fbank(self, filename, filename2=None, idx=None):
    #     # mixup
    #     if filename2 == None:
    #         waveform, sr = torchaudio.load(filename)
    #         waveform = waveform - waveform.mean()
    #     # mixup
    #     else:
    #         waveform1, sr = torchaudio.load(filename)
    #         waveform2, _ = torchaudio.load(filename2)

    #         waveform1 = waveform1 - waveform1.mean()
    #         waveform2 = waveform2 - waveform2.mean()

    #         if waveform1.shape[1] != waveform2.shape[1]:
    #             if waveform1.shape[1] > waveform2.shape[1]:
    #                 # padding
    #                 temp_wav = torch.zeros(1, waveform1.shape[1])
    #                 temp_wav[0, 0:waveform2.shape[1]] = waveform2
    #                 waveform2 = temp_wav
    #             else:
    #                 # cutting
    #                 waveform2 = waveform2[0, 0:waveform1.shape[1]]

    #         # sample lambda from uniform distribution
    #         #mix_lambda = random.random()
    #         # sample lambda from beta distribtion
    #         mix_lambda = np.random.beta(10, 10)

    #         mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
    #         waveform = mix_waveform - mix_waveform.mean()
        

    #     ## yb: align ##
    #     if waveform.shape[1] > 16000*(self.opt.audio_length+0.1):
    #         sample_indx = np.linspace(0, waveform.shape[1] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
    #         waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)]
    #     ## align end ##


    #     # if self.opt.vis_encoder_type == 'vit':
    #     #     fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
    #     #     # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=512, dither=0.0, frame_shift=1)
    #     # elif self.opt.vis_encoder_type == 'swin':
    #     fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

    #     ########### ------> very important: audio normalized
    #     fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
    #     ### <--------
    #     if self.opt.vis_encoder_type == 'vit':
    #         target_length = int(1024 * (1/10)) ## for audioset: 10s
    #     elif self.opt.vis_encoder_type == 'swin':
    #         target_length = 192 ## yb: overwrite for swin

    #     # target_length = 512 ## 5s
    #     # target_length = 256 ## 2.5s
    #     n_frames = fbank.shape[0]

    #     p = target_length - n_frames

    #     # cut and pad
    #     if p > 0:
    #         m = torch.nn.ZeroPad2d((0, 0, 0, p))
    #         fbank = m(fbank)
    #     elif p < 0:
    #         fbank = fbank[0:target_length, :]

    #     if filename2 == None:
    #         return fbank, 0
    #     else:
    #         return fbank, mix_lambda

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                num_acts = len(value['annotations'])
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(value['annotations']):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    labels[idx] = label_dict[act['label']]
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_item['id'] + self.file_ext)
        with np.load(filename) as data:
            feats = data['feats'].astype(np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        
        
        if self.use_addtional_feats:
            audio_length = 1
            
            
            file_pre = "/mnt/welles/scratch/datasets/epic-audio/"
            additional_file_name = file_pre + video_item['id'] + '.wav'
            waveform, sample_rate = torchaudio.load(additional_file_name)
            if waveform.shape[0] == 2:  # Check if stereo
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            
            window_length_secs = 0.025  # For example, 25ms
            hop_length_secs = 0.010    # For example, 10ms

            # Calculate window and hop lengths in samples
            window_length_samples = int(round(sample_rate * window_length_secs))
            hop_length_samples = int(round(sample_rate * hop_length_secs))

            # Calculate FFT length (nearest power of 2)
            fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
            mel_transformer = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=fft_length, win_length=window_length_samples, hop_length=hop_length_samples, n_mels=64)
            mel_spectrogram = mel_transformer(waveform)

            # Convert to log scale
            log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)  # To avoid log(0)
            
            mean = torch.mean(log_mel_spectrogram)
            std = torch.std(log_mel_spectrogram)
            log_mel_spectrogram = torch.divide(log_mel_spectrogram-mean, std+1e-9)
            
            audio_feats = []
            max_length = 208
            
            for feat_id in range(feats.shape[-1]):
                # Calculate the start and end time of the video feature
                video_feature_duration = 32 / 30  # Duration of video feature in seconds
                feature_start_time = feat_id * (16 / 30)  # Start time of the video feature in seconds
                feature_end_time = feature_start_time + video_feature_duration  # End time of the video feature

                # Adjust start and end times for the desired audio segment duration
                audio_segment_start_time = feature_start_time - (audio_length / 2)
                audio_segment_end_time = feature_end_time + (audio_length / 2)

                # Convert times to mel-spectrogram indices
                # Here, we need to consider the frame rate of the mel-spectrogram
                hop_length = mel_transformer.hop_length
                start_index = math.floor(audio_segment_start_time * sample_rate / hop_length)
                end_index = math.ceil(audio_segment_end_time * sample_rate / hop_length)

                # Ensure indices are within bounds
                start_index = max(start_index, 0)
                end_index = min(end_index, log_mel_spectrogram.shape[-1])

                # Extract the corresponding audio segment from the mel-spectrogram
                audio_segment = log_mel_spectrogram[:, :, start_index:end_index]
                if audio_segment.shape[-1] < max_length:
                    padding_size = max_length - audio_segment.shape[-1]
                    audio_segment = torch.nn.functional.pad(audio_segment, (0, padding_size), "constant", 0)
                audio_feats.append(audio_segment)
            additional_feats = torch.stack(audio_feats).squeeze()
            additional_feats = additional_feats.permute(1, 2, 0)

            # if samples.shape[0] > 16000*(audio_length+0.1):
            #     sample_indx = np.linspace(0, samples.shape[0] -16000*(audio_length+0.1), num=feats.shape[-1], dtype=int)
            #     samples = samples[sample_indx[idx]:sample_indx[idx]+int(16000*audio_length)]

            # else:
            #     # repeat in case audio is too short
            #     samples = np.tile(samples,int(self.audio_length))[:int(16000*self.audio_length)]

            # samples[samples > 1.] = 1.
            # samples[samples < -1.] = -1.
   
            
            # path_folder = '/CV/datasets/thumos14/pose_heatmap'
            # additional_feats = np.load(
            #     os.path.join(self.additional_feat_folder, additional_file_name))  # T, kpt_cls, height, width
            # additional_feats = torch.from_numpy(additional_feats).to(torch.float32)

            # if additional_feats.shape[0] == 1:
            #     additional_feats = additional_feats.squeeze(0)

            # additional_feats = additional_feats.flatten(1)  # T, cls, height* width
            # additional_feats = additional_feats.transpose(0, 1)  # cls*height* width, T
            # additional_feats = \
            #     F.interpolate(additional_feats[None], feats.shape[-1], mode='linear', align_corners=True)[0]
        else:
            additional_feats = None
        
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps']- 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames,
                     'additional_feats': additional_feats,}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict
