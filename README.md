# SPATE-model
Spatiotemporal embeddings
This project provides the code of SPATE (Spatiotemporal embeddings), a model for learning vector space embeddings for representing a spatiotemporal cells, using Flickr tags and structured information extracted from various scientific resources. The structured information can be of two forms: numerical and categorical information.

This implementation uses Python and TensorFlow (a "library for numerical computation using data flow graphs" by Google). You can use the sample of the data to run the code, however, the results reported in the paper are based on larger datasets.


# How to run this code?

import SPATE

model = SPATE.Model(embedding_size=50, learning_rate=0.5, batch_size=1024, scaling_factor= , cat_weight= )

model.fit(region_len,NF_len+cat_len+vocab_len)

model.train(num_epochs=30,Txt_file,NF_file,Cat_file)
