import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

import torch.nn.functional as F
import torchvision.transforms as transforms
from basicsr.archs.dfdnet_arch import DFDNet
from basicsr.utils import imwrite, tensor2img

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, enhance_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm = SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    # while ret:
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            detect_results = detect_model.get(frame,crop_size)

            if detect_results is not None:
                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                frame_align_crop_list = detect_results[0]
                frame_mat_list = detect_results[1]
                frame_part_loc_list = detect_results[2]

                swap_result_list = []
                swap_enh_result_list = []
                frame_align_crop_tenor_list = []
                for frame_align_crop, frame_part_loc in zip(frame_align_crop_list, frame_part_loc_list):
                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB))[None,...].cuda()

                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    #print(swap_result)

                    try:
                        with torch.no_grad():
                            cropped_face = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(swap_result).unsqueeze(0)
                           
                                
                            #cropped_face = F.interpolate(cropped_face, size=(512, 512), mode='nearest')

                            # cv2.imwrite('./inner_results/cropped_upscale/' + 'cropped_{:0>7d}.jpg'.format(frame_index), 
                            #     cropped_face.clone().cpu().detach().numpy()[0].transpose((1, 2, 0)) * 255)



                            swap_enhanced_result = enhance_model(cropped_face, frame_part_loc)[0]
                            
                            #swap_enhanced_result = F.interpolate(swap_enhanced_result.unsqueeze(0), size=(224, 224), mode='area')[0]

                            swap_enhanced_result = swap_enhanced_result.float().detach().cpu().clamp_(-1, 1)
                            swap_enhanced_result = (swap_enhanced_result + 1) / 2

                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f'DFDNet inference fail: {e}')
                                            
                    swap_img = swap_result.cpu().detach().numpy().transpose((1, 2, 0)) * 255
                    swap_enh_img = swap_enhanced_result.cpu().detach().numpy().transpose((1, 2, 0)) * 255

                    #cv2.imwrite('./inner_results/swapped/' + 'swapped_{:0>7d}.jpg'.format(frame_index), swap_img)
                    #cv2.imwrite('./inner_results/enhanced/' + 'enhanced_{:0>7d}.jpg'.format(frame_index), swap_enh_img)

                    swap_result_list.append(swap_enhanced_result)
                    # swap_enh_result_list.append(swap_enhanced_result)
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                reverse2wholeimage(frame_align_crop_tenor_list, swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                    os.path.join(temp_results_dir, 'frame_{:0>7d}.png'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.png'.format(frame_index)), frame)
        else:
            break

    video.release()

    path = os.path.join(temp_results_dir,'*.png')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    clips.write_videofile(save_path,audio_codec='rawvideo')

