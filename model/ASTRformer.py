import torch
import torch.nn as nn


class ASTRformer(nn.Module):
    def __init__(self,seq_len,num_nodes,in_feature,d_model,pred_len,args,**kwargs):
        super().__init__()
        self.E_as=nn.Parameter(torch.randn(d_model,num_nodes,seq_len,device='cuda')) #空间Embedding
        self.E_at = nn.Parameter(torch.randn(d_model,num_nodes,seq_len, device='cuda')) #时间Embedding
        self.embedding=nn.Conv2d(in_feature,d_model,kernel_size=1)
        self.conv_s=nn.Conv2d(2*d_model,d_model,kernel_size=1)
        self.conv_t = nn.Conv2d(2 * d_model, d_model, kernel_size=1)
        self.conv=nn.Sequential(nn.Conv2d(2 * d_model, 2*d_model, kernel_size=1),
                      nn.ReLU(),nn.Conv2d(2 * d_model, d_model, kernel_size=1))
        self.TTransformer=Transformer(embed_size=d_model, heads=8, time_num=seq_len, dropout=0, forward_expansion=8)
        self.STransformer = Transformer(embed_size=d_model, heads=8, time_num=num_nodes, dropout=0, forward_expansion=8)
        self.output_time=nn.Conv2d(seq_len,pred_len,kernel_size=1)
        self.output_dim = nn.Conv2d(d_model,in_feature, kernel_size=1)


    def forward(self,x,adj,**kwargs):
        x=self.embedding(x) # (B,D,N,L)
        B,D,N,L=x.shape
        z_s=self.conv_s(torch.cat([x,self.E_as.unsqueeze(0).repeat(B,1,1,1)],dim=1))
        z_t = self.conv_s(torch.cat([x, self.E_at.unsqueeze(0).repeat(B,1,1,1)], dim=1))
        z_st=self.conv(torch.cat([z_s,z_t],dim=1)) # (B,D,N,L)
        out_t=self.TTransformer(z_st.permute(0,2,3,1)) #(B,N,T,D)
        out_s = self.STransformer(out_t.transpose(1,2))#(B,T,N,D)
        output=self.output_time(out_s)#(B,O,N,D)
        output=output.transpose(1,-1)#(B,D,N,O)
        output=torch.relu(output)
        output=self.output_dim(output)#(B,C,N,O)
        return output


class Selfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Selfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = self.embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # q, k, v:[B, N, T, C]
        B, N, T, C = query.shape

        # q, k, v:[B, N,T,heads, per_dim]
        keys = key.reshape(B, N, T, self.heads, self.per_dim)
        queries = query.reshape(B, N, T, self.heads, self.per_dim)
        values = value.reshape(B, N, T, self.heads, self.per_dim)

        keys = self.keys(keys)
        values = self.values(values)
        queries = self.queries(queries)

        # compute temperal self-attention
        attnscore = torch.einsum("bnqhd, bnkhd->bnqkh", (queries, keys))  # [B,N, T, T, heads]
        attention = torch.softmax(attnscore / (self.embed_size ** (1/2)), dim=2)

        out = torch.einsum("bnqkh, bnkhd->bnqhd", (attention, values)) # [B, N, T, heads, per_dim]
        out = out.reshape(B,N, T, self.embed_size)
        out = self.fc(out)

        return out


class Transformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(Transformer, self).__init__()
        # Temporal embedding One hot
        self.time_num = time_num
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding

        self.attention = Selfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query):
        # q, k, v：[B, N, T, C]
        B, N, T, C = query.shape

        D_T = self.temporal_embedding(torch.arange(0, T).to(query.device))  # temporal embedding选用nn.Embedding
        D_T = D_T.expand(B, N, T, C)

        # TTransformer
        x = D_T + query
        attention = self.attention(x, x, x)
        M_t = self.dropout(self.norm1(attention + x))
        feedforward = self.feed_forward(M_t)
        U_t = self.dropout(self.norm2(M_t + feedforward))

        out = U_t + x + M_t # 残差连接

        return out

