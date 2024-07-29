import torch
import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from gensim.models import Word2Vec
import text_encoding

import numpy as np

class ArtDatasetKGM(data.Dataset):

    def __init__(self, args_dict, set, att2i, att_name, transform=None, append='False', clusters=15, k=100):
        """
        Args:
            args_dict: parameters dictionary
            set: 'train', 'val', 'test'
            att2i: attribute vocabulary
            att_name: either 'type', 'school', 'time', or 'author'
            transform: data transform
        """

        self.args_dict = args_dict
        self.set = set
        self.embedds = args_dict.embedds
        # Load Data + Graph Embeddings
        self.graphEmb = []
        
        if self.embedds == 'graph':
            self.graphEm = Word2Vec.load(os.path.join(args_dict.dir_data, args_dict.graph_embs))
        elif self.embedds == 'bow':
            self.chosen_coded_semart_train, self.chosen_coded_semart_val, self.chosen_coded_semart_test = \
            text_encoding.bow_load_train_text_corpus(args_dict.dir_dataset, append='append', k=k)

            self.chosen_coded_semart_train = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_train, clusters=clusters)
            self.chosen_coded_semart_val = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_val, clusters=clusters)
            self.chosen_coded_semart_test = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_test, clusters=clusters)
            
            pd.DataFrame(self.chosen_coded_semart_train).to_csv('train_semart_fcm_'+str(clusters)+'.csv')
            pd.DataFrame(self.chosen_coded_semart_val).to_csv('val_semart_fcm_'+str(clusters)+'.csv')
            pd.DataFrame(self.chosen_coded_semart_test).to_csv('test_semart_fcm_'+str(clusters)+'.csv')
        
        elif self.embedds == 'kmeans':
            self.chosen_coded_semart_train, self.chosen_coded_semart_val, self.chosen_coded_semart_test = \
            text_encoding.bow_load_train_text_corpus(args_dict.dir_dataset, append='append', k=k)

            self.chosen_coded_semart_train = text_encoding.kmeans_coded_context(
                self.chosen_coded_semart_train, clusters=clusters)
            self.chosen_coded_semart_val = text_encoding.kmeans_coded_context(
                self.chosen_coded_semart_val, clusters=clusters)
            self.chosen_coded_semart_test = text_encoding.kmeans_coded_context(
                self.chosen_coded_semart_test, clusters=clusters)
            
            pd.DataFrame(self.chosen_coded_semart_train).to_csv('train_semart_fcm_'+str(clusters)+'.csv')
            pd.DataFrame(self.chosen_coded_semart_val).to_csv('val_semart_fcm_'+str(clusters)+'.csv')
            pd.DataFrame(self.chosen_coded_semart_test).to_csv('test_semart_fcm_'+str(clusters)+'.csv')
            
        elif self.embedds == 'frbc':
            self.chosen_coded_semart_train = pd.read_csv('Data/rule_embds_train.csv', index_col=0).values
            self.chosen_coded_semart_val = pd.read_csv('Data/rule_embds_val.csv', index_col=0).values
            self.chosen_coded_semart_test = pd.read_csv('Data/rule_embds_test.csv', index_col=0).values 
       

        elif self.embedds == 'tfidf':
            self.chosen_coded_semart_train, self.chosen_coded_semart_val, self.chosen_coded_semart_test = \
            text_encoding.tf_idf_load_train_text_corpus(args_dict.dir_dataset, append='append', k=k)

            self.chosen_coded_semart_train = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_train, clusters=clusters)
            self.chosen_coded_semart_val = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_val, clusters=clusters)
            self.chosen_coded_semart_test = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_test, clusters=clusters)
        
        elif self.embedds == 'clip':
            self.chosen_coded_semart_train, self.chosen_coded_semart_val, self.chosen_coded_semart_test = \
            text_encoding.clip_load_train_text_corpus(args_dict.dir_dataset, append='append', k=k)



        if self.set == 'train':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
        elif self.set == 'val':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
        elif self.set == 'test':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)


        df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')

        self.imagefolder = os.path.join(args_dict.dir_dataset, args_dict.dir_images)
        self.transform = transform
        self.att2i = att2i
        self.imageurls = list(df['IMAGE_FILE'])
        if att_name == 'type':
            self.att = list(df['TYPE'])
        elif att_name == 'school':
            self.att = list(df['SCHOOL'])
        elif att_name == 'time':
            self.att = list(df['TIMEFRAME'])
        elif att_name == 'author':
            self.att = list(df['AUTHOR'])

    def __len__(self):
        return len(self.imageurls)


    def class_from_name(self, vocab, name):

        if name in vocab:
            idclass= vocab[name]
        else:
            idclass = vocab['UNK']

        return idclass


    def __getitem__(self, index):
        """Returns data sample as a pair (image, description)."""

        # Load image & apply transformation
        imagepath = self.imagefolder + self.imageurls[index]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Attribute class
        idclass = self.class_from_name(self.att2i, self.att[index])

        ''' # Attribute class
        type_idclass = self.class_from_name(self.type_vocab, self.type[index])
        school_idclass = self.class_from_name(self.school_vocab, self.school[index])
        time_idclass = self.class_from_name(self.time_vocab, self.time[index])
        author_idclass = self.class_from_name(self.author_vocab, self.author[index])'''

        
        # Graph embedding (only training samples)
        if self.set == 'train':
            if self.embedds == 'graph':
                graph_emb = self.graphEm.wv[self.imageurls[index]]
            else:
                graph_emb = self.chosen_coded_semart_train[index, :]

            graph_emb = torch.FloatTensor(graph_emb)

            if self.args_dict.att == 'all':
                return [image], [type_idclass, school_idclass, time_idclass, author_idclass, graph_emb], self.imageurls[index]
            else:
                return [image], [idclass, graph_emb], self.imageurls[index]


        elif self.set == 'val':
            if self.embedds == 'graph':
                graph_emb = self.graphEm.wv[self.imageurls[index]]
            else:
                graph_emb = self.chosen_coded_semart_val[index, :]

            graph_emb = torch.FloatTensor(graph_emb)

            if self.args_dict.att == 'all':
                return [image], [type_idclass, school_idclass, time_idclass, author_idclass, graph_emb], self.imageurls[index]
            else:
                return [image], [idclass, graph_emb], self.imageurls[index]
        
        elif self.set == 'test':
            if self.embedds == 'graph':
                    graph_emb = np.random.rand(128,1)#; self.graphEm.wv[self.imageurls[index]]
            else:
                graph_emb = self.chosen_coded_semart_test[index, :]

            graph_emb = torch.FloatTensor(graph_emb)
            
            if self.args_dict.att == 'all':
                return [image], [type_idclass, school_idclass, time_idclass, author_idclass, graph_emb], self.imageurls[index]
            else:
                return [image], [idclass, graph_emb], self.imageurls[index]