from __future__ import division

import numpy as np
import torch
from torchvision import transforms

from model_kgm import KGM, KGM_append, get_gradcam, GradCamKGM
from dataloader_kgm import ArtDatasetKGM
from attributes import load_att_class

import pandas as pd
#from model_gcn import GCN, 
NODE2VEC_OUTPUT = 128


def extract_grad_cam_features(visual_model, data, target_var, args_dict, batch_idx, im_names):
    print(len(target_var))
    print(len(data))
    
    # for ix, image in enumerate(data):
    #     ix_0 = int(target_var[0][ix].cpu().numpy())
    #     ix_1 = int(target_var[1][ix].cpu().numpy())
    #     ix_2 = int(target_var[2][ix].cpu().numpy())
    #     ix_3 = int(target_var[3][ix].cpu().numpy())
    #     grad_cam_image = 0.25 * get_gradcam(visual_model, image, ix_0, 0) + \
    #                     0.25 * get_gradcam(visual_model, image, ix_1, 1) + \
    #                     0.25 * get_gradcam(visual_model, image, ix_2, 2) + \
    #                     0.25 * get_gradcam(visual_model, image, ix_3, 3)
    
    for ix, image in enumerate(data):
        
        ix_0 = int(target_var[ix].cpu().numpy())
        grad_cam_image = get_gradcam(visual_model, image, ix_0, 0)
        
        if ix == 0:
            grad_cams = torch.zeros((data.shape[0], 1, grad_cam_image.shape[0], grad_cam_image.shape[1]))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            grad_cams = grad_cams.to(device)

        grad_cams[ix] = grad_cam_image

    for jx in range(grad_cams.shape[0]):
        grad_cam = grad_cams[jx, 0, :, :].detach().cpu().numpy()
        pd.DataFrame(grad_cam).to_csv('./GradCams/' + im_names[jx] + '.csv', index=False)

    

def test_knowledgegraph(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    if args_dict.att == 'type':
        att2i = type2idx
        num_classes = len(type2idx)
    elif args_dict.att == 'school':
        att2i = school2idx
        num_classes = len(school2idx)
    elif args_dict.att == 'time':
        att2i = time2idx
        num_classes = len(time2idx)
    elif args_dict.att == 'author':
        att2i = author2idx
        num_classes = len(author2idx)
    elif args_dict.att == 'all':
        att2i = [type2idx, school2idx, time2idx, author2idx]
        num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]

    N_CLUSTERS = args_dict.clusters
    symbol_task = args_dict.symbol_task
    # Define model
   
    # Define model
    if args_dict.embedds == 'graph':
        if args_dict.append != 'append':
            print('Case 1')
            model = GradCamKGM(len(att2i))
            #model = KGM(len(att2i), end_dim=N_CLUSTERS, model=args_dict.architecture, multi_task=args_dict.att=='all')
        else:
            print('Case 2')
            model = KGM_append(len(att2i), end_dim=N_CLUSTERS, model=args_dict.architecture, multi_task=args_dict.att=='all')
    else:
        if args_dict.append != 'append':
            print('Correct new models')
            #model = KGM(num_classes, end_dim=N_CLUSTERS, model=args_dict.architecture, multi_task=args_dict.att=='all')
            model = GradCamKGM(num_classes, end_dim=N_CLUSTERS)
        else:
            print('Case 3')
            model = KGM_append(len(att2i), end_dim=N_CLUSTERS, model=args_dict.architecture)

    if torch.cuda.is_available():#args_dict.use_gpu:
        model.cuda()
    
    # Load best model

    print("=> loading checkpoint '{}'".format(args_dict.model_path))
    checkpoint = torch.load(args_dict.model_path, encoding='latin1')
    args_dict.start_epoch = checkpoint['epoch']
    # Create a random batch
    dummy_batch = torch.randn(1, 3, 224, 224).cuda()
    model(dummy_batch)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args_dict.model_path, checkpoint['epoch']))
    '''except RuntimeError as e:
        print(e)
        print('No checkpoint available')
        args_dict.start_epoch = 0'''


    # Data transformation for test
    test_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for test
    if torch.cuda.is_available():
  
        test_loader = torch.utils.data.DataLoader(
            ArtDatasetKGM(args_dict, set='test', att2i=att2i, att_name=args_dict.att, transform=test_transforms, clusters=128),
            batch_size=args_dict.batch_size, shuffle=False, pin_memory=(not args_dict.no_cuda), num_workers=args_dict.workers)
            
      
    else:
        test_loader = torch.utils.data.DataLoader(
            ArtDatasetKGM(args_dict, set='test', att2i=att2i, att_name=args_dict.att, transform=test_transforms, clusters=128),
            batch_size=args_dict.batch_size, shuffle=False, pin_memory=False,
            num_workers=args_dict.workers)

    # Switch to evaluation mode & compute test samples embeddings
    model.eval()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # full_imgs = []
    for i, (input, target, im_names) in enumerate(test_loader):
        # Inputs to Variable type
        if args_dict.model == 'kgm':
            target, embd = target

        input_var = list()
        for j in range(len(input)):
            if torch.cuda.is_available():
                input_var.append(torch.autograd.Variable(input[j]).cuda())
            else:
                input_var.append(torch.autograd.Variable(input[j]))

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            if args_dict.att == 'all':
                target[j] = torch.tensor(np.array(target[j], dtype=np.uint8))
                if torch.cuda.is_available():
                    target[j] = target[j].cuda(non_blocking=True)
                target_var.append(torch.autograd.Variable(target[j]))
            else:
                target_var = list()
                for j in range(len(target)):
                    target[j] = torch.tensor(np.array(target[j], dtype=np.uint8))

                    if torch.cuda.is_available():
                        target[j] = target[j].cuda(non_blocking=True)

                    target_var.append(torch.autograd.Variable(target[j]))

 #       target_var.append(torch.autograd.Variable(target[j]))
 #       print(target_var)
        
        # Output of the model
        if args_dict.symbol_task:
            output = model(input_var[0])
        elif args_dict.append == 'append':
            output = model((input_var[0], target[1]))
            # feat_cache = model.features((input_var[0], target[1]))   
        elif args_dict.att == 'all':
            pred_type, pred_school, pred_tf, pred_author, _ = model(input_var[0]) 
            # feat_cache = model.features(input_var[0]).detach().cpu().numpy()
        elif args_dict.embedds == 'graph' or args_dict.embedds == 'bow' or args_dict.embedds == 'kmeans':
            output, context_output = model(input_var[0])
            # feat_cache = model.features(input_var[0])
        else:
            output = model(input_var[0])
            # feat_cache = model.features(input_var[0])   

        #outsoftmax = torch.nn.functional.softmax(output[0])
        
        if args_dict.att == 'all':
            conf, predicted_type = torch.max(pred_type, 1)
            conf, predicted_school = torch.max(pred_school, 1)
            conf, predicted_time = torch.max(pred_tf, 1)
            conf, predicted_author = torch.max(pred_author, 1)

            out_type = predicted_type.data.cpu().numpy()
            out_school = predicted_school.data.cpu().numpy()
            out_time = predicted_time.data.cpu().numpy()
            out_author = predicted_author.data.cpu().numpy()

            label_type = target[0].cpu().numpy()
            label_school = target[1].cpu().numpy()
            label_time = target[2].cpu().numpy()
            label_author = target[3].cpu().numpy()

        else:
            # print(output)
            conf, predicted = torch.max(output, 1)

            out_actual = predicted.data.cpu().numpy()
            target = [int(trt) for trt in target]
            label_actual = np.array(target) # .cpu().numpy()

        if i==0:
            if args_dict.att == 'all':
                out_type = predicted_type.data.cpu().numpy()
                out_school = predicted_school.data.cpu().numpy()
                out_time = predicted_time.data.cpu().numpy()
                out_author = predicted_author.data.cpu().numpy()

                label_type = target[0].cpu().numpy()
                label_school = target[1].cpu().numpy()
                label_time = target[2].cpu().numpy()
                label_author = target[3].cpu().numpy()

                scores = conf.data.cpu().numpy()
            else:
                out = out_actual
                labels = label_actual # .cpu().numpy()
            # logits = output.data.cpu().numpy()

        else:
            if args_dict.att == 'all':
                out_type = np.concatenate((out_type,predicted_type.data.cpu().numpy()),axis=0)
                out_school = np.concatenate((out_school,predicted_school.data.cpu().numpy()),axis=0)
                out_time = np.concatenate((out_time,predicted_time.data.cpu().numpy()),axis=0)
                out_author = np.concatenate((out_author,predicted_author.data.cpu().numpy()),axis=0)

                label_type = np.concatenate((label_type,target[0].cpu().numpy()),axis=0)
                label_school = np.concatenate((label_school,target[1].cpu().numpy()),axis=0)
                label_time = np.concatenate((label_time,target[2].cpu().numpy()),axis=0)
                label_author = np.concatenate((label_author,target[3].cpu().numpy()),axis=0)

                scores = np.concatenate((scores, conf.data.cpu().numpy()), axis=0)
            else:
                out = np.concatenate((out,out_actual),axis=0)
                labels = np.concatenate((labels, label_actual),axis=0)

            # logits = np.concatenate((logits, output.data.cpu().numpy()), axis=0)
        
        
        extract_grad_cam_features(model, input_var[0], target_var, args_dict, i, im_names)
        # print(features_matrix[actual_index:actual_index+args_dict.batch_size].shape, feat_cache.shape)
        # features_matrix[actual_index:actual_index+feat_cache.shape[0], :] = feat_cache
        # actual_index += feat_cache.shape[0]
        # full_imgs.append(input_var[0].cpu().numpy())

    # pd.DataFrame(features_matrix, index=full_imgs).to_csv('./DeepFeatures/test_x_' + str(args_dict.att) + '_' + str(args_dict.embedds) + '.csv', index=True)
    # Compute Accuracy
    
        # Map labels to numbers
        print(out.shape, labels.shape)
        acc = np.mean(np.equal(out, labels))
        print('Model %s\tTest Accuracy %.03f' % (args_dict.model_path, acc))
        

def run_test(args_dict):

    if args_dict.model == 'kgm':
        test_knowledgegraph(args_dict)
    else:
        assert False, 'Incorrect model type: ' + args_dict.model

