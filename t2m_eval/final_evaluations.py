from datetime import datetime
import numpy as np
import torch
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from motion_loaders.model_motion_loaders import get_motion_loader, get_motion_loader_for_chois_eval 
from utils.get_opt import get_opt
from utils.metrics import *
from networks.evaluator_wrapper import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
# from scripts.motion_process import *
from utils import paramUtil
from utils.utils import *

from options.train_options import TrainTexMotMatchOptions

from os.path import join as pjoin

from vis_skeleton_motion import show3Dpose_animation 
from data.omomo_dataset import CanoObjectTrajDataset 
from utils.word_vectorizer import WordVectorizer, POS_enumerator
import pickle as pkl
import random

def plot_t2m(data, save_dir, ds):
    # data: BS X 2 X T X D 
    num_steps = data.shape[2]
   
    # data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        # joint_data = data[i][:, :, :24*3].reshape(-1, num_steps, 24, 3) # 2 X T X 24 X 3
        # joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # joint = ds.de_normalize_jpos_min_max(joint_data) # 2 X T X 24 X 3 
        joint_data = data[i][:, :, :24*3] # 2 X T X 72 
        joint = ds.de_normalize_jpos_mean_std(joint_data) # 2 X T X 72 
        joint = joint.reshape(-1, num_steps, 24, 3) # 2 X T X 24 X 3 
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        # plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)
        show3Dpose_animation(joint.detach().cpu().numpy(), ds.parents, save_path) 

# def plot_t2m(data, save_dir, captions):
#     data = gt_dataset.inv_transform(data)
#     # print(ep_curves.shape)
#     for i, (caption, joint_data) in enumerate(zip(captions, data)):
#         joint = recover_from_ric(torch.from_numpy(joint_data).float(), wrapper_opt.joints_num).numpy()
#         save_path = pjoin(save_dir, '%02d.mp4'%(i))
#         plot_3d_motion(save_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
#         # print(ep_curve.shape)

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                # import pdb; pdb.set_trace()
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3) # zyd
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        print(motion_embeddings.shape)
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict

def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    replication_times = 1 
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file, test_name, model_name, eval_motion_loaders):
    with open(log_file, 'w') as f:
        # all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
        #                            'R_precision': OrderedDict({}),
        #                            'FID': OrderedDict({}),
        #                            'Diversity': OrderedDict({}),
        #                            'MultiModality': OrderedDict({})})
        
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}), # zyd
                                   })
        replication_times = 1 
        for replication in range(replication_times):
            motion_loaders = {}
            # mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                # mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            # zyd
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)
            # zyd

            # print(f'Time: {datetime.now()}')
            # print(f'Time: {datetime.now()}', file=f, flush=True)
            # mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            # zyd
            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            # zyd

            # for key, item in mm_score_dict.items():
            #     if key not in all_metrics['MultiModality']:
            #         all_metrics['MultiModality'][key] = [item]
            #     else:
            #         all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            # print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    # print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
            
        for model_name, values in fid_score_dict.items():
            if model_name == 'ground truth' or model_name == 'ground_truth':
                continue
            epoch = model_name.split('_')[-1]
            pkl_path = os.path.join('/cpfs04/shared/sport/zouyude/code/lingo-release/code/results', test_name, 'metrics_'+str(epoch)+'.pkl')
            with open(pkl_path, 'rb') as f:
                metrics_data = pkl.load(f)
                metrics_data['summary']['FID'] = values
                metrics_data['summary']['FID_std'] = 0.0
            with open(pkl_path, 'wb') as f:
                pkl.dump(metrics_data, f)

        for model_name, values in R_precision_dict.items():
            if model_name == 'ground truth' or model_name == 'ground_truth':
                continue
            epoch = model_name.split('_')[-1]
            pkl_path = os.path.join('/cpfs04/shared/sport/zouyude/code/lingo-release/code/results', test_name, 'metrics_'+str(epoch)+'.pkl')
            with open(pkl_path, 'rb') as f:
                metrics_data = pkl.load(f)
                metrics_data['summary']['R_precision'] = values[-1]
                metrics_data['summary']['R_precision_std'] = 0.0
            with open(pkl_path, 'wb') as f:
                pkl.dump(metrics_data, f)

        for model_name, values in div_score_dict.items():
            if model_name == 'ground truth' or model_name == 'ground_truth':
                continue
            epoch = model_name.split('_')[-1]
            pkl_path = os.path.join('/cpfs04/shared/sport/zouyude/code/lingo-release/code/results', test_name, 'metrics_'+str(epoch)+'.pkl')
            with open(pkl_path, 'rb') as f:
                metrics_data = pkl.load(f)
                metrics_data['summary']['Diversity'] = values
                metrics_data['summary']['Diversity_std'] = 0.0
            with open(pkl_path, 'wb') as f:
                pkl.dump(metrics_data, f)

def check_vis(save_dir):
    w_vectorizer = WordVectorizer('/move/u/jiamanli/github/text-to-motion/glove_840B', 'our_vab')

    data_root_folder = "/move/u/jiamanli/datasets/semantic_manip/processed_data"
    val_dataset = CanoObjectTrajDataset(train=False, data_root_folder=data_root_folder, \
                word_vectorizer=w_vectorizer) 
    
    motion_loaders = {}
    for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
        motion_loader = motion_loader_getter()
        motion_loaders[motion_loader_name] = motion_loader
       
    motion_loaders['ground_truth'] = gt_loader
    for motion_loader_name, motion_loader in motion_loaders.items():
        for idx, batch in enumerate(motion_loader):
            if not (idx % 4 == 0):
                continue 

            word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, tokens = batch
            motions = motions[:, :m_lens[0]] # BS X T X 72 
            # plot_t2m(motions.cpu().numpy(), save_path, captions)
            print('-----%d-----'%idx)
            print(captions)
            print(tokens)
            print(sent_lens)
            print(m_lens)

            ani_save_path = pjoin(save_dir, 'animation', '%02d'%(idx))
            os.makedirs(ani_save_path, exist_ok=True)
           
            # data = gt_dataset.inv_transform(motions[0])
            # print(ep_curves.shape)
            
            # save_path = pjoin(save_dir, '%02d.mp4' % (idx))
            plot_t2m(motions[:8, None], pjoin(ani_save_path, '%s' % (motion_loader_name)),
                          val_dataset)


if __name__ == '__main__':
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    # dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD01/opt.txt'
    root_dir = '/cpfs04/shared/sport/zouyude/code/lingo-release/code/t2m_results_48'
    # test_list = ['chois_2_lo_occ_const', 'chois_2_lo_occ', 'chois_3_lo_occ_const', 'chois_3_lo_occ']
    # test_list = ['chois_lo_const_all', 'chois_2_lo_const_all']
    # test_list = ['chois_lo_const_all_410', 'chois_2_lo_const_all_410']
    # test_list = ['chois_523_500_rdp', 'chois_523_500_rdp_w_obj', 'chois_523_500_rdp_w_scale', 'chois_525_500_dp_wo_obj']
    # test_list = ['chois_530_temp_dp_new', 'chois_530_temp_dp_new_w_g_1_2', 'chois_530_temp_dp_new_w_g_obj_only', 'chois_530_temp_dp_new_w_guidance']
    # test_list = ['chois_523_wo_dp_temp_w_obj', 'chois_523_wo_dp_wo_obj', 'chois_525_dp_dual_wo_obj']
    # test_list = ['chois_625_dual', 'chois_623_temp', 'chois_623_temp_all', 'chois_623_pe_old', 'chois_623_pe_x0']
    # test_list = ['chois_623_temp_all_cm', 'chois_623_temp_cm', 'chois_623_pe_old_cm']
    # test_list = ['chois_626_pe_x0_012', 'chois_626_pe_x0_2_012', 'chois_626_pe_x0_012_cm', 'chois_626_pe_x0_2_012_cm']
    # test_list = ['chois_623_temp_all_cm_F', 'chois_623_temp_cm_F', 'chois_626_pe_x0_012_cm_F', 'chois_626_pe_x0_2_012_cm_F']
    # test_list = ['chois_623_pe_old_cm_F']
    # test_list = ['chois_626_pe_x0_cm_1_0-1_F', 'chois_626_pe_x0_cm_1_0-01_F', 'chois_626_pe_x0_cm_1_0-5_F',
    #              'chois_626_pe_x0_cm_1_1_F', 'chois_626_pe_x0_cm_3_0-1_F', 'chois_626_pe_x0_cm_3_0-5_F', 'chois_626_pe_x0_cm_3_1_F']
    # test_list = ['chois_626_pe_x0_012_-1', 'chois_626_pe_x0_012_0', 'chois_626_pe_x0_012_1', 
    #              'chois_626_pe_x0_012_2', 'chois_626_pe_x0_012_3', 'chois_626_pe_x0_012_4',
    #              'chois_626_pe_x0_012_5', 'chois_626_pe_x0_012_6', 'chois_626_pe_x0_012_6-5']
    # test_list = ['chois_626_pe_x0_012_cm_cfg_-1', 'chois_626_pe_x0_012_cm_cfg_0', 'chois_626_pe_x0_012_cm_cfg_1', 
    #              'chois_626_pe_x0_012_cm_cfg_2', 'chois_626_pe_x0_012_cm_cfg_3', 'chois_626_pe_x0_012_cm_cfg_4',
    #              'chois_626_pe_x0_012_cm_cfg_5', 'chois_626_pe_x0_012_cm_cfg_6']
    # test_list = ['chois_626_pe_x0_cm_3_10_F', 'chois_705_pe_x0_cm_3_10_F', 'chois_705_pe_x0_cm_3_20_F', 'chois_705_pe_x0_cm_3_50_F']
    # test_list = ['chois_626_pe_x0_012', 'chois_704_pe_x0_pen_0-1', 'chois_704_pe_x0_pen_0-5', 'chois_704_pe_x0_pen_1', 'chois_704_pe_x0_pen_10']
    # test_list = ['chois_716_pe_x0_20', 'chois_716_pe_x0_50', 'chois_716_pe_x0_100']
    # test_list = ['chois_722_pe_x0_o0']
    # test_list = ['chois_716_pe_x0_10_cm', 'chois_716_pe_x0_o0_10_cm']
    # test_list = ['chois_724_pe_x0_cm_pen_0', 'chois_724_pe_x0_cm_pen_0-1', 'chois_724_pe_x0_cm_pen_1',
    #              'chois_724_pe_x0_cm_pen_0-01', 'chois_724_pe_x0_cm_pen_0-001', 'chois_716_pe_x0_o0_10_cm']
    # test_list = ['chois_716_pe_x0_o0_50_-1', 'chois_716_pe_x0_o0_50_0', 'chois_716_pe_x0_o0_50_1',
                #  'chois_716_pe_x0_o0_50_2', 'chois_716_pe_x0_o0_50_3', 'chois_716_pe_x0_o0_50_4']
    # test_list = ['chois_731_pe_x0_o0_cm', 'chois_731_pe_x0_o0_cm_cfg_0', 'test', 
    #              'chois_731_pe_x0_o0_cm_cfg_1', 'chois_731_pe_x0_o0_cm_cfg_2', 'chois_731_pe_x0_o0_cm_cfg_-1_new']
    # test_list = ['chois_731_pe_x0_o0_cm_cfg_1', 'chois_731_pe_x0_o0_cm_cfg_5_1', 'chois_731_pe_x0_o0_cm_cfg_50_1']
    # test_list = ['chois_806_pe_x0_o0_cm_cfg_vel_0-01_1', 'chois_806_pe_x0_o0_cm_cfg_vel_0-1_1', 
    #              'chois_806_pe_x0_o0_cm_cfg_vel_1_1', 'chois_806_pe_x0_o0_cm_cfg_vel_10_1']
    # test_list = ['chois_731_pe_x0_o0_cm_cfg_fk_1', 'chois_731_pe_x0_o0_cm_cfg_fk_2', 'chois_731_pe_x0_o0_cm_cfg_fk_5', 'chois_731_pe_x0_o0_cm_cfg_fk_10']
    # test_list = ['chois_716_pe_x0_o0_50_only_obj']
    # test_list = ['chois_806_pe_x0_o0_cm_cfg_vel_10_1_only_obj', 'chois_806_pe_x0_o0_cm_cfg_vel_1_1_only_obj', 'chois_731_pe_x0_o0_cm_cfg_50_1_only_obj']
    # test_list = ['chois_809_pe_x0_o0_voxel_0', 'chois_809_pe_x0_o0_voxel_1', 'chois_716_pe_x0_o0_50_1']
    # test_list = ['lingo_lo_810_10_only_obj']
    # test_list = ['chois_806_pe_x0_o0_cm_cfg_vel_0-1_1_only_obj', 'chois_806_pe_x0_o0_cm_cfg_vel_1_1_only_obj', 'chois_806_pe_x0_o0_cm_cfg_vel_10_1_only_obj']
    # test_list = ['chois_806_pe_x0_o0_cm_cfg_vel_fk_1_only_obj', 'chois_806_pe_x0_o0_cm_cfg_vel_fk_2_only_obj',  'chois_731_pe_x0_o0_cm_cfg_0-5_1_only_obj']
    # test_list = ['chois_820_pe_x0_voxel_3_w1', 'chois_820_pe_x0_voxel_2_w1', 'chois_820_pe_x0_voxel_1_w1', 'chois_820_pe_x0_voxel_0_w1']
    # test_list = ['chois_809_pe_x0_o0_voxel_0_w1', 'chois_813_pe_x0_voxel_3_w1']
    # test_list = ['chois_813_pe_x0_voxel_3_cm_500_re']
    # test_list = ['chois_909_pe_x0_voxel_3_cm_500']
    test_list = ['lingo_lo_909_0-1']
    for test_name in test_list:
        exp_dir = os.path.join(root_dir, test_name)
        model_list = os.listdir(exp_dir)
        model_list.sort()
        for model_idx in range(len(model_list)-1,-1,-1):
            model_name = model_list[model_idx]
            epoch = model_name.split('_')[-1]
            pkl_path = os.path.join('/cpfs04/shared/sport/zouyude/code/lingo-release/code/results', test_name, 'metrics_'+str(epoch)+'.pkl')
            try:
                with open(pkl_path, 'rb') as f:
                    metrics_data = pkl.load(f)
                if metrics_data['summary']['FID'] is not None:
                    print(f"已经计算过FID指标,跳过 {pkl_path}")
                    continue
            except:
                pass
            eval_motion_loaders = {}
            eval_motion_loaders[model_name] = lambda: get_motion_loader_for_chois_eval(
                os.path.join(exp_dir, model_name),
                batch_size,
            )

            batch_size = 32
            diversity_times = 300 

            gt_loader = get_motion_loader_for_chois_eval(
                    '/cpfs04/shared/sport/zouyude/code/lingo-release/code/t2m_results_48/gt',
                    batch_size)
            # gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
            
            parser = TrainTexMotMatchOptions()
            wrapper_opt = parser.parse()

            device_id = 0
            wrapper_opt.device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(device_id)

            # wrapper_opt = get_opt(dataset_opt_path, device)
            eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

            log_file = './t2m_evaluation_chois_single_window_chois.log'
            evaluation(log_file, test_name, model_name, eval_motion_loaders)

            # save_vis_folder = "/move/u/jiamanli/eccv2024_chois/check_fid_eval_res"
            # check_vis(save_vis_folder)
    