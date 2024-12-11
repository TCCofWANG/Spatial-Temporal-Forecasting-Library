from torch import nn
import torch
from layers.STIDGCN_related import *

class STIDGCN(nn.Module):
    def __init__(
        self, input_dim, num_nodes, channels, granularity,input_len,output_len,  points_per_hour, dropout=0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.input_len=input_len
        self.points_per_hour = points_per_hour
        diffusion_step = 1

        self.Temb = TemporalEmbedding(granularity, channels, points_per_hour, num_nodes)

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )

        self.tree = IDGCN_Tree(
            channels=channels*2,
            diffusion_step=diffusion_step,
            num_nodes=self.num_nodes,
            dropout=dropout,
        )

        self.glu = GLU(channels*2, dropout)

        self.regression_layer = nn.Conv2d(
            channels*2, input_dim, kernel_size=(1, 1)
        )
        self.out=nn.Linear(self.input_len,self.output_len)


    def forward(self, input,adj,**kwargs):
        seqs_time=kwargs.get("seqs_time")
        targets_time=kwargs.get("targets_time")
        x = input
        # Encoder
        # Data Embedding
        time_emb = self.Temb(seqs_time)  # output:(B,C,N,L)
        x = torch.cat([self.start_conv(x)] + [time_emb], dim=1)
        # IDGCN_Tree
        x = self.tree(x)
        # Decoder
        gcn = self.glu(x) + x
        prediction = self.regression_layer(F.relu(gcn))
        prediction=self.out(prediction)
        return prediction