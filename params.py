import argparse
import ast

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='gen_graph_dataset', type=str, help='Mode (train | test)')
    parser.add_argument('--model', default='kgm', type=str, help='Model (mtl | kgm). mlt for multitask learning model. kgm for knowledge graph model.' )
    parser.add_argument('--att', default='time', type=str, help='Attribute classifier (type | school | time | author| all) (only kgm model).')

    # Directories
    parser.add_argument('--dir_data', default='Data')
    parser.add_argument('--dir_dataset', default='../SemArt/')
    parser.add_argument('--dir_images', default='Images/')
    parser.add_argument('--dir_model', default='Models/')
    parser.add_argument('--visual_cache', default='Embeds/VisReduce/')

    # Files
    parser.add_argument('--csvtrain', default='semart_train.csv', help='Training set data file')
    parser.add_argument('--csvval', default='semart_val.csv', help='Dataset val data file')
    parser.add_argument('--csvtest', default='semart_test.csv', help='Dataset test data file')
    parser.add_argument('--vocab_type', default='type2ind.pckl', help='Type classes file')
    parser.add_argument('--vocab_school', default='school2ind.pckl', help='Author classes file')
    parser.add_argument('--vocab_time', default='time2ind.pckl', help='Timeframe classes file')
    parser.add_argument('--vocab_author', default='author2ind.pckl', help='Author classes file')

    # Training opts
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--nepochs', default=300, type=int)

    
    # KGM model
    parser.add_argument('--graph_embs', default='semart-artgraph-node2vec.model')
    parser.add_argument('--lambda_c', default=0.9, type=float)
    parser.add_argument('--lambda_e', default=0.1, type=float)
    parser.add_argument('--embedds', default='graph', type=str)
    parser.add_argument('--append', default='gradient')

    # FCM model
    parser.add_argument('--k', default=100, type=int)
    parser.add_argument('--clusters', default=128, type=int)

    # GCN model
    parser.add_argument('--feature_matrix', default='Data/feature_train_128_semart.csv', type=str)
    parser.add_argument('--val_feature_matrix', default='Data/feature_val_128_semart.csv', type=str)
    parser.add_argument('--test_feature_matrix', default='Data/feature_test_128_semart.csv', type=str)

    parser.add_argument('--edge_list_train', default='Data/kg_semart.csv', type=str)
    parser.add_argument('--edge_list_val', default='Data/kg_semart_val.csv', type=str)
    parser.add_argument('--edge_list_test', default='Data/kg_semart_test.csv', type=str)
    
    # MTL model
    parser.add_argument('--architecture', default='resnet', type=str)

    # Test
    parser.add_argument('--model_path', default='Models/best-kgm-time-model.pth.tar', type=str)
    parser.add_argument('--no_cuda', action='store_true')

    #Symbol task
    parser.add_argument('--symbol_task', default=False, type=bool)
    parser.add_argument('--targets', type=ast.literal_eval)

    # Grad cam
    parser.add_argument('--grad_cam_model_path', default='Models/grad_cam_lenet.pth.tar', type=str)
    parser.add_argument('--grad_cam_images_path', default='./GradCams/', type=str)

    parser.add_argument('--base', default='context', type=str)
    return parser