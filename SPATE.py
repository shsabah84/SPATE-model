from __future__ import division
from random import shuffle
import tensorflow as tf
import numpy as np
import sys
import psutil
import gc
import datetime
import os


class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class Model():
    def __init__(self, embedding_size, batch_size=512, learning_rate=0.05, alpha=0.5, beta=0.5):

        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta= beta
        self.__embeddings = None
        
    def fit(self, region_len, vocab_len, NF_len, cat_len,mnth_len,latlon_len):
        self.__fit(region_len, vocab_len, NF_len, cat_len, mnth_len,latlon_len)
        self.__build_graph()
        
    def __fit(self, region_len, vocab_len, NF_len, cat_len,mnth_len,latlon_len):         
        self.vocab_size=vocab_len
        self.region_size=region_len
        self.Nfeatures_size=NF_len
        self.cat_size=cat_len
        self.mnth_size=mnth_len
        self.latlon_size=latlon_len
        
        
    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device("/cpu:0"):
          
            alpha = tf.constant([self.alpha], dtype=tf.float32,
                                         name="alpha")
            
            beta = tf.constant([self.beta], dtype=tf.float32,
                                         name="beta")
            
            
            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")#region id
            
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")# all context id
            
            self.__cooccurrence_count = tf.placeholder(tf.float16, shape=[self.batch_size],
                                                       name="cooccurrence_count")#ppmi or nf value

            self.__NF_input = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                  name="numerical_features")#mask
            
            self.__tags_input = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="tags")#mask
            
            self.__cat_input = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="categories")#mask
            
            self.__mnth_input = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="month")#mask
            
            self.__latlon_input = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="latlon")#mask
            
            
            tags_weight = 1-(2*alpha)-(2*beta)

            focal_embeddings = tf.Variable(
                tf.random_uniform([self.region_size, self.embedding_size], 1.0, -1.0),name="focal_embeddings")#embedding_size=dimentions
            
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0), name="context_embeddings")
            
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")#biase of all context features
            mnth_bias = tf.Variable(tf.random_uniform([2], 1.0, -1.0),
                                         name="month_biases")
            
            T = tf.Variable(tf.random_uniform([self.embedding_size,2], 1.0, -1.0),name="months_projection_matrix")
            
            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)#biase of both tags and nf
            
            
            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding),1)#dot product
            self.__emp=embedding_product
            distance_expr = tf.square(tf.add_n([#((wi*wj)+bj-log)**2
                embedding_product,
                context_bias,
                tf.negative(tf.to_float(self.__cooccurrence_count))]))#change float16 to 32
            self.__disexp=distance_expr

            ###### Glove (tegs) part
            tags_single_losses = tf.multiply(self.__tags_input, distance_expr)#tags loss*mask
            self.__tags_loss = tf.multiply(tf.reduce_sum(tags_single_losses),tags_weight)# distance * model weight *taginput
        
            ###### NF part
            NF_single_losses = tf.multiply(self.__NF_input, distance_expr)#nf loss*mask
            self.__NF_loss = tf.multiply(tf.reduce_sum(NF_single_losses),alpha)#distance* 1-model weight  *nfinput   

            ###### categorical features part
            dcat=tf.reduce_sum(tf.square(tf.subtract(focal_embedding, context_embedding)),1)#distance
            catembedding_product = tf.multiply(tf.to_float(self.__cooccurrence_count),dcat)#weighted euclidean distance
            cat_single_losses = tf.multiply(self.__cat_input, catembedding_product)#*mask
            self.__cat_loss = tf.multiply(tf.reduce_sum(cat_single_losses),alpha)#similarity*weight

            
            ###### latlon part
            latlon_single_losses = tf.multiply(self.__latlon_input, distance_expr)#nf loss*mask
            self.__latlon_loss = tf.multiply(tf.reduce_sum(latlon_single_losses),beta)#distance* 1-model weight  *nfinput       
            

            ###### temporal features part
            projection=tf.matmul(focal_embedding,T)#multiply py the projection matrix
            c=tf.cos([self.__cooccurrence_count])
            s=tf.sin([self.__cooccurrence_count])
            w=tf.transpose(tf.to_float(tf.concat([c,s], 0)))
            similarity1=tf.add(tf.subtract(projection,w),mnth_bias)#euclidean distance
            similarity=tf.reduce_sum(tf.square(similarity1),1)
            mnth_single_losses = tf.multiply(self.__mnth_input, similarity)#*mask
            self.__mnth_loss = tf.multiply(tf.reduce_sum(mnth_single_losses),beta)#similarity*weight

            
            self.__total_loss = tf.add_n([self.__tags_loss, self.__NF_loss, self.__cat_loss, self.__mnth_loss,self.__latlon_loss])#sumij
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)##########optimizer
            self.__combined_embeddings2 = (focal_embeddings)

    def train(self, num_epochs, corpus, NF, cat,log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
        print('loading data......................')
        cooccurrences =[]
        
        f_NF=open(NF,'r')        
        cooccurrences = cooccurrences +[(int(l.split()[0]),int(l.split()[1]),float(l.split()[2]),0,1,0,0,0)
                           for l in f_NF]
        del(f_NF)
        print('length after the NF',len(cooccurrences))
        
        f_cat=open(cat,'r')
        cooccurrences = cooccurrences +[(int(l.split()[0]),int(l.split()[1])+self.Nfeatures_size,float(l.split()[2]),0,0,1,0,0)#I just dublicate the index of context for input data shape only
                   for l in f_cat]
        del(f_cat)
        print('length after the cat',len(cooccurrences))

        f_mnth=open(mnth,'r')
        cooccurrences = cooccurrences +[(int(l.split()[0]),int(l.split()[1])+self.Nfeatures_size+self.cat_size,float(((int(l.split()[1])+1)*(2*np.pi))/12),0,0,0,1,0)
                   for l in f_mnth]
        del(f_mnth)
        print('length after the mnth',len(cooccurrences))
        
        f_latlon=open(latlon,'r')
        cooccurrences = cooccurrences +[(int(l.split()[0]),int(l.split()[1])+self.Nfeatures_size+self.cat_size+self.mnth_size,float(l.split()[2]),0,0,0,0,1)
                                        for l in f_latlon]
        del(f_latlon)
        print('length after the latlon',len(cooccurrences))
        
        f_words=open(corpus,'r')
        cooccurrences = cooccurrences +[(int(l.split()[0]),int(l.split()[1])+self.Nfeatures_size+self.cat_size+self.mnth_size+self.latlon_size,float(l.split()[2]),1,0,0,0,0)
                                        for l in f_words]
        del(f_words)
        print('length after the tags',len(cooccurrences))
        
        shuffle(cooccurrences)
        	
        gc.collect()
        with tf.Session(graph=self.__graph) as session:#####start the session(main)
            print('start initializing global variables..........')
            tf.global_variables_initializer().run()###initialize the variables
            batches = self.__prepare_batches(cooccurrences)######shuffle and divided data into batches
            del(cooccurrences)
            
            for epoch in range(num_epochs):#iterations
                print('the begining of itr '+str(epoch)+':\n')
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts,i1 ,i2, i3 ,i4,i5= batch######count=cooccurrence
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {#fill the holder with batch data
                        self.__focal_input: i_s,#id
                        self.__context_input: j_s,#id
                        self.__cooccurrence_count: counts,#cooccur
                        self.__tags_input:i1,
                        self.__NF_input:i2,
                        self.__cat_input:i3,
                        self.__mnth_input:i4,
                        self.__latlon_input:i5}
                    __,tloss,tagsl,nfl,catl,mls,latlonls=session.run([self.__optimizer,self.__total_loss,self.__tags_loss,self.__NF_loss,
                                                                      self.__cat_loss,self.__mnth_loss,self.__latlon_loss],
                                                                     feed_dict=feed_dict)##########run optimization

                print('tags loss',tagsl)
                print('NF loss', nfl)
                print('Cat loss',catl)
                print('Months loss',mls)
                print('Coordinates loss',latlonls)
                print('Total loss',tloss)
                                    
            self.__embeddings = self.__combined_embeddings2.eval()
            print('priniting on a file.........')
            np.savetxt('em_1eq300.txt',self.__embeddings,delimiter=' ',newline='\n',fmt='%.7f')
            current_time = datetime.datetime.now()
            print("Printed!!!!!!!!!!!!!!!! at {:%H:%M}".format(current_time)) 
        
                        
    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self,cooccurrences):
        i_indices, j_indices, counts, i1, i2, i3,i4,i5= zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts,i1 ,i2,i3,i4,i5))
    

def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)
    del(sequences)
    gc.collect()

def _device_for_node(n):    
    return "/cpu:0"

