# IMDb-Sentiment-Analysis-using-DAN-and-GRU

In the IMDB sentiment analysis task, there are three datasets with different number of movie reviews (5k, 10k and 15k). Each review has been pre-label with ’positive’ or ’negative’ sentiment class based on review contents.


To convert text into numerical values, we use to pre-trained GloVe word vectors to perform word embeddings. Then two deep learning models - Deep Averaging Network (DAN) and Gated Recurret United (GRU) are trained to perform the classification tasks.
Deep Averaging Network contains following steps:
1. Take the vector average of the embeddings associated with an input sequence of tokens.
2. Pass that average through one or more feed-forward layer. Within the layers, we apply dropout with probability of 0.2 and ReLu activation function.
3. Perform linear classification on the final layers representation.

Gated Recurrent Unit is the younger sibling of the more popular Long short-term memory (LSTM) network and also a type of RNN. One drawback of DAN model is that the ordering of words were not considered. GRU are able to retain long-term dependencies in sequential data and address the gradient vanishing issue in vanilla RNN. Figure 2 shows that the structure of a GRU unit. It contains two gates:
1. Reset Gate
2. Update gate
