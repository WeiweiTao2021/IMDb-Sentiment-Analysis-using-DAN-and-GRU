# std lib imports
from typing import Dict

# external libs
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

class SequenceToVector(nn.Module):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self, input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``torch.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``torch.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : torch.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : torch.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(DanSequenceToVector, self).__init__(input_dim)

        # TODO(students): start
        self.dan = nn.Sequential()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.dan.add_module("linear_"+str(i), nn.Linear(input_dim, input_dim))
                self.dan.add_module("dropout_"+str(i), nn.Dropout(dropout))
                self.dan.add_module("relu_"+str(i), nn.ReLU())
            else:
                self.dan.add_module("linear_"+str(i), nn.Linear(input_dim, input_dim))
                self.dan.add_module("dropout_"+str(i), nn.Dropout(dropout))
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:

        # TODO(students): start
        ## take the average of the input sequence
        x = torch.squeeze(torch.bmm(sequence_mask.reshape(sequence_mask.shape[0], 1, sequence_mask.shape[1]), vector_sequence))/sequence_mask.sum(axis = 1).reshape(sequence_mask.shape[0], 1)

        temp = []
        for layer in self.dan:
            if training or (not isinstance(layer, torch.nn.Dropout)):
                if isinstance(layer, torch.nn.Linear):
                    temp.append(x)
                x = layer(x)
        
        temp.append(x)
        temp.pop(0)

        combined_vector = x
        layer_representations = torch.stack(temp)

        ##print("The layer size of DAN is :", layer_representations.shape)

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(GruSequenceToVector, self).__init__(input_dim)

        # TODO(students): start; Added dropout
        self.gru = nn.GRU(input_dim, input_dim, num_layers, batch_first=True, dropout=dropout)
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        
        # TODO(students): start
        embedded = nn.utils.rnn.pack_padded_sequence(vector_sequence, lengths=sequence_mask.sum(axis = 1), batch_first = True, enforce_sorted = False)
        _, layer_representations = self.gru(embedded)

        combined_vector = layer_representations[-1]

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
