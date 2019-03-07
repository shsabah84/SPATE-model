import SPATE
model = SPATE.Model(embedding_size=50, learning_rate=0.5, batch_size=1024, alpha=0.04, beta=0.45)
print('model generated')

#input data files
Tags_file = 'Tags.txt' #the textual data file must be in the form of  “region_id  term_id  value”
Numerical_Features_file = 'Numerical_Features.txt' #the numerical data file must be in the form of  “region_id  feature_id  value”
Categorical_Features_file = 'Categorical_Features.txt' #the categorical data file must be in the form of “region_id  feature_id ”
Spatial_Features_file = 'Spatial_Features.txt'#the numerical data file must be in the form of  “region_id  feature_id  value”
Temporal_Features_file='Temporal_Features.txt' #the numerical data file must be in the form of  “region_id  month_id  month_corresponding_point”
    
region_len=200000 #number of regions or entities
NF_len=7 #number of numerical features
cat_len=180 #number of categories
vocab_len=100000 #number of terms in the corpus
month_len=12# 12 months
latlon_len=2#2 spatial coordinates

context_len=NF_len+cat_len+vocab_len+month_len+latlon_len

model.fit(region_len,context_len,NF_len,cat_len,month_len,latlon_len)
num_epochs=30 #number of iterations
model.train(num_epochs,Tags_file,Numerical_Features_file,Categorical_Features_file,Temporal_Features_file,Spatial_Features_file)
#the embedding vectors will be saved in EGEL.txt file which contains the vectors for all the regions each at a line


    
