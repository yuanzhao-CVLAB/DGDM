# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import copy
import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import  math
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb,control):
        for layer in self:
            if isinstance(layer, TimestepBlock) or  isinstance(layer,AttentionBlock):
                x = layer(x, emb,control)

            else:
                x = layer(x)
        return x


class PositionalEmbedding(nn.Module):
    # PositionalEmbedding
    """
    Computes Positional Embedding of the timestep
    """

    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels
        if use_conv:
            # downsamples by 1/2
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            assert in_channels == out_channels
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        # uses upsample then conv to avoid checkerboard artifacts
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        x = F.interpolate(x.float(), scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, in_channels, n_heads=1, n_head_channels=-1,group_num=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(group_num, self.in_channels)
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert (
                    in_channels % n_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {n_head_channels}"
            self.num_heads = in_channels // n_head_channels

        # query, key, value for attention
        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads,in_channels)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))
        self.gait_unit = nn.Linear(384,in_channels*2)

    def forward(self, x, time=None,control=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.to_qkv(self.norm(x))
        h = self.attention(qkv,control=control)
        h = self.proj_out(h)
        class_features = control.get("features",None)
        if class_features is not None:
            class_features = self.gait_unit(class_features).unsqueeze(-1).chunk(2,dim = 1)
            h = h*F.sigmoid(class_features[0])+class_features[1]
        return (x + h).reshape(b, c, *spatial)



class Routed_VQ(nn.Module):
    def __init__(self, embedding_dim,num_embed,num_cls = 15):
        super(Routed_VQ, self).__init__()
        self.vq = nn.ModuleList([VectorQuantizerEMA(embedding_dim, num_embed) for i in range(num_cls)])

    def forward(self, x,cls):
        cond_list=[]
        for c, d in zip(cls, x):
            cond_list.append(self.vq[c](d.unsqueeze(0)))
        x = torch.cat(cond_list, dim=0)

        return x

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.loss = 0
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        # 用于存储损失
        self.quantization_loss = torch.tensor(0.).cuda()
    def get_loss(self):
        loss = self.quantization_loss
        self.quantization_loss = torch.tensor(0.).cuda()
        return loss
    def get_quantized(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self.embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        self.quantization_loss += loss.mean()
        # print(self.quantization_loss )
        # convert quantized from BHWC -> BCHW
        return  quantized.permute(0, 3, 1, 2).contiguous()



class DAF(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DAF, self).__init__()
        self.conv_prototype = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            BatchNorm(out_channels)
        )
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            BatchNorm(out_channels)
        )

    def forward(self, x, prototype, discrepancy):
        shortcut = x
        weight = torch.sigmoid(discrepancy)
        x1 = self.conv_x((1 - weight) * x)
        x2 = self.conv_prototype(weight * prototype)

        return x1+x2 + shortcut
import torch
import torch.nn as nn
import torch.nn.functional as F

class LiteBottleneck(nn.Module):
    """
    A minimal residual bottleneck for anomaly-detection backbones.
    Args:
        in_ch  (int): input/output channels
        stride (int): 1 (default) or 2 for optional down-sampling
        dilation (int): receptive-field size, default 1
    """
    def __init__(self, in_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        mid = in_ch // 4                    # 4× squeeze
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.conv2 = nn.Conv2d(
            mid, mid, 3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.conv3 = nn.Conv2d(mid, in_ch, 1, bias=False)
        self.norm  = nn.GroupNorm(in_ch//4, in_ch)
        # self.se = nn.AdaptiveAvgPool2d(1)  # ← 如需 SE，解注释并乘以通道权重
        self.act   = nn.SiLU(inplace=True)

        # 如果下采样，则给残差分支也做平均池以匹配尺寸
        self.downsample = (
            nn.AvgPool2d(2) if stride == 2 else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.act(out)

        out = self.conv2(out)
        out = self.act(out)

        out = self.conv3(out)
        # 如需通道注意力： out = self.se(out) * out
        out = self.norm(out)                # 统一放在末尾，省一层

        return self.act(out + identity)



class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads,in_channels):
        super().__init__()
        self.n_heads = n_heads
        self.gconv = nn.Sequential(
                    nn.Conv2d(3 , in_channels, 4,stride=4),
                    GroupNorm32(in_channels//8, in_channels),
                    nn.SiLU(),
            nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1, stride=1),
                )

        self.patch_smooth_conv = nn.Sequential(
                    nn.Conv2d(1 , in_channels, 4,stride=4),
                )

        self.score_conv= nn.Sequential(
                    nn.Conv2d(in_channels , in_channels, 3,padding=1,stride=1),
                )
        self.bottleneck = LiteBottleneck(in_channels,1)
        self.conv_vq = nn.Conv2d(in_channels , in_channels, 2,stride=2)

        self.in_channels = in_channels
        self.routed_VQ = Routed_VQ(8,in_channels)#routed_VQ
        self.daf = DAF(in_channels,in_channels)
    def forward(self, qkv, time=None,control=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
                "bct,bcs->bts", q * scale, k * scale
                )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if control.get("pred_x0") is not None and control["cond"]:


            a = torch.einsum("bts,bcs->bct", weight, v)
            a= a.reshape(bs, -1, length)
            HW = int(math.sqrt(length))
            a  = a.unflatten(-1,(HW ,HW))
            ####prototype generation###
            x_0 = self.gconv(control.get("augmented_image"))
            cls_idx = control["classes"].argmax(-1)
            prototype = x_0
            prototype = self.bottleneck(prototype)
            prototype = self.routed_VQ(prototype, cls_idx)
            prototype = self.conv_vq(prototype)
            prototype =F.interpolate(prototype,size = (HW,HW),mode = "bilinear")

            diff_map = torch.norm(control.get("augmented_image") - control.get("pred_x0"), dim=1, keepdim=True)  # [B, 1, H, W]
            activate  = self.patch_smooth_conv(diff_map)
            activate  = F.interpolate(activate, size=(HW, HW), mode="bilinear")
            activate = self.score_conv(activate)
            a = self.daf(a,prototype,activate).flatten(-2,-1)

        else:
            a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)
# class QKVAttention(nn.Module):
#     """
#     A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
#     """
#
#     def __init__(self, n_heads,in_channels):
#         super().__init__()
#         self.n_heads = n_heads
#         self.gconv = nn.Sequential(
#                     nn.Conv2d(3 , in_channels, 4,stride=4),
#                     GroupNorm32(in_channels//8, in_channels),
#                     nn.SiLU(),
#             nn.Conv2d(in_channels, in_channels,kernel_size=3,padding=1, stride=1),
#                 )
#
#         self.bottleneck = LiteBottleneck(in_channels,1)
#
#         self.patch_smooth_conv = nn.Sequential(
#                     nn.Conv2d(1, in_channels, 4, stride=4),
#                     nn.Conv2d(in_channels, in_channels, 3,padding=1,stride=1),
#                 )
#
#
#
#
#         self.conv_vq = nn.Conv2d(in_channels , in_channels, 2,stride=2)
#         self.in_channels = in_channels
#         self.routed_vq = Routed_VQ(8,in_channels)
#         self.daf = DAF(in_channels,in_channels)
#     def forward(self, qkv, time=None,control=None):
#         """
#         Apply QKV attention.
#         :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = torch.einsum(
#                 "bct,bcs->bts", q * scale, k * scale
#                 )  # More stable with f16 than dividing afterwards
#         weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
#
#         a = torch.einsum("bts,bcs->bct", weight, v)
#         if control.get("pred_x0") is not None and control["cond"]:
#             a= a.reshape(bs, -1, length)
#             HW = int(math.sqrt(length))
#             a  = a.unflatten(-1,(HW ,HW))
#             #prototype generation
#             prototype = self.gconv(control.get("augmented_image"))
#             cls_idx = control["classes"].argmax(-1)
#             prototype = self.bottleneck(prototype)
#             prototype = self.routed_vq(prototype, cls_idx)
#             prototype = self.conv_vq(prototype)
#             prototype = F.interpolate(prototype, size=(HW, HW), mode="bilinear")
#
#
#             ##discrepancy-aware mechanism
#             diff_map = torch.norm(control.get("augmented_image") - control.get("pred_x0"), dim=1, keepdim=True)  # [B, 1, H, W]
#
#             diff_map  = self.patch_smooth_conv[0](diff_map)
#             diff_map = F.interpolate(diff_map,size = (HW,HW),mode = "bilinear")
#             diff_map = self.patch_smooth_conv[1](diff_map)
#             weight = torch.sigmoid(diff_map)
#             a = self.daf(a,prototype,weight).flatten(-2,-1)
#         return a.reshape(bs, -1, length)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, num_heads=8, dropout_rate=0.1):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.depth = query_dim // num_heads

        self.Wq = nn.Linear(query_dim, query_dim, bias=False)
        self.Wk = nn.Linear(key_value_dim, query_dim, bias=False)
        self.Wv = nn.Linear(key_value_dim, query_dim, bias=False)

        self.dense = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth).
        # Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    #N,L,D
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.split_heads(self.Wq(query), batch_size)
        key = self.split_heads(self.Wk(key), batch_size)
        value = self.split_heads(self.Wv(value), batch_size)

        # Scaled dot-product attention
        matmul_qk = torch.matmul(query.transpose(-2, -1), key)
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Attention weights
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Output
        output = torch.matmul(attention_weights, value.transpose(-2, -1)).transpose(-2, -1)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.query_dim)

        # Final linear layer
        output = self.dense(output)

        return output


class ResBlock(TimestepBlock):
    def __init__(
            self,
            in_channels,
            time_embed_dim,
            dropout,
            out_channels=None,
            use_conv=False,
            up=False,
            down=False,
            control=None,
            img_size = None,
            group_num = 32,
            ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_layers = nn.Sequential(
                GroupNorm32(group_num, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
                )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels, False)
            self.x_upd = Upsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.embed_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels)
                )
        self.out_layers = nn.Sequential(
                GroupNorm32(group_num, out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

        # self.class_embedding = nn.Embedding(control["classes"],out_channels)
        # self.edge_conv =  nn.Sequential(
        #     nn.SiLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Conv2d(1, 1, 3, stride=1,padding=1),
        #     *[nn.Sequential(
        #     GroupNorm32(1, 1),
        #     nn.SiLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Conv2d(1, 1, 3, stride=2,padding=1)
        # ) for _ in range(int(np.log2(256//img_size)))])
        # self.edge_embedding = nn.Embedding(control["classes"],img_size*img_size)
        # self.edge_alpha_conv = nn.Sequential(
        #     GroupNorm32(1, 1),
        #     nn.SiLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Conv2d(1, 1, 3, stride=1,padding=1),
        #     nn.Sigmoid()
        # )
        # self.class_linear1 = nn.Sequential(GroupNorm32(group_num, time_embed_dim),nn.Linear(time_embed_dim,out_channels*2),)

        # self.gate_unit =nn.Sequential(GroupNorm32(group_num, time_embed_dim), nn.Linear(time_embed_dim, out_channels * 2), nn.SiLU(),
        #               nn.Dropout(p=dropout), )
        self.gate_unit = nn.Linear(time_embed_dim,out_channels*2)
    def forward(self, x, time_embed,control):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.embed_layers(time_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # if control.get("edges") is not None:
        #     edge = self.edge_conv(control["edges"])
        #     edge_mask = self.edge_embedding(control["class_idx"])
        #     alpha = self.edge_alpha_conv(edge)
        #     edge = alpha*edge+(1-alpha)*edge_mask.view(edge.shape)
        if control.get("features") is not None  and control["cond"]:
            features =  self.gate_unit(control.get("features")).unsqueeze(-1).unsqueeze(-1).chunk(2,dim=1)
            prompt = F.sigmoid(features[0]) * features[1]
            h =h + prompt

        # classes_emb = self.class_embedding(control["class_idx"]).unsqueeze(-1).unsqueeze(-1)

        h = h + emb_out #+ classes_emb #+edge
        h = self.out_layers(h)
        return self.skip_connection(x) + h

def get_unet(args,classes):
    control_config = {"classes":classes}
    model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args["dropout"],
                      n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],in_channels= args["channels"],control = control_config )
    return model

    # image_size=args['img_size'][0]
    # if args['channel_mults'] == "":
    #     if image_size == 512:
    #         channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    #     elif image_size == 256:
    #         channel_mult = (1, 1, 2, 2, 4, 4)
    #     elif image_size == 128:
    #         channel_mult = (1, 1, 2, 3, 4)
    #     elif image_size == 64:
    #         channel_mult = (1, 2, 3, 4)
    #     else:
    #         raise ValueError(f"unsupported image size: {image_size}")
    # model =  CDM_UNET(
    #     image_size=image_size,
    #     in_channels=args["channels"],
    #     model_channels=args["num_head_channels"],
    #     out_channels=6 ,
    #     num_res_blocks=2,
    #     attention_resolutions=[8,16,32],
    #     dropout=args["dropout"],
    #     channel_mult=channel_mult,
    #     num_classes=None,
    #     use_checkpoint=False,
    #     use_fp16=True,
    #     num_heads=args["num_heads"],
    #     num_head_channels=args["num_head_channels"],
    #     num_heads_upsample=-1,
    #     use_scale_shift_norm=True,
    #     resblock_updown=True,
    #     use_new_attention_order=False,
    # )
    # res = model.load_state_dict(
    #     dist_util.load_state_dict(args["checkpoint_path"], map_location="cpu"),strict=False
    # )
    #
    # print(res)
    # return model
class UNetModel(nn.Module):
    # UNet model
    def __init__(
            self,
            img_size,
            base_channels,
            conv_resample=True,
            n_heads=1,
            n_head_channels=-1,
            channel_mults="",
            num_res_blocks=2,
            dropout=0,
            attention_resolutions="32,16,8",
            biggan_updown=True,
            in_channels=1,
            control=None,
            ):
        self.dtype = torch.float32
        super().__init__()

        if channel_mults == "":
            if img_size == 512:
                channel_mults = (0.5, 1, 1, 2, 2, 4, 4)
            elif img_size == 256:
                # channel_mults = (1, 1, 2, 3, 4)
                channel_mults = (1, 1, 2, 3, 4)
            elif img_size == 128:
                channel_mults = (1, 1, 2, 3, 4)
            elif img_size == 64:
                channel_mults = (1, 2, 3, 4)
            elif img_size == 32:
                channel_mults = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {img_size}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(img_size // int(res))

        self.image_size = img_size
        self.in_channels = in_channels
        self.model_channels = base_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample

        self.dtype = torch.float32
        self.num_heads = n_heads
        self.num_head_channels = n_head_channels

        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
                PositionalEmbedding(base_channels, 1),
                nn.Linear(base_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
                )

        ch = int(channel_mults[0] * base_channels)
        self.down = nn.ModuleList(
                [TimestepEmbedSequential(nn.Conv2d(self.in_channels, base_channels, 3, padding=1))]
                )
        channels = [ch]
        ds = 1

        cur_size = img_size

        for i, mult in enumerate(channel_mults):
            # out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        out_channels=base_channels * mult,
                        dropout=dropout,
                        control = control,
                    img_size=cur_size,
                        )]
                ch = base_channels * mult
                # channels.append(ch)

                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels,
                                    )
                            )
                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)
            if i != len(channel_mults) - 1:
                out_channels = ch
                cur_size//=2
                self.down.append(
                        TimestepEmbedSequential(
                                ResBlock(
                                        ch,
                                        time_embed_dim=time_embed_dim,
                                        out_channels=out_channels,
                                        dropout=dropout,
                                        down=True,
                                        control = control,
                                        img_size=cur_size
                                        )
                                if biggan_updown
                                else
                                 Downsample(ch, conv_resample, out_channels=out_channels)
                                )
                        )
                ds *= 2
                ch = out_channels
                channels.append(ch)
        self.residual_blocks = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                control=control,
                img_size=cur_size
            ),
            AttentionBlock(
                ch,
                n_heads=n_heads,
                n_head_channels=n_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                down=True,
                control=control,
                img_size=cur_size
            ),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.cls_head = nn.Linear(ch, control["classes"])
        self.middle = TimestepEmbedSequential(
                ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        control = control,
                        img_size=cur_size
                        ),
                AttentionBlock(
                        ch,
                        n_heads=n_heads,
                        n_head_channels=n_head_channels,
                        ),
                ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        control = control,
                        img_size=cur_size
                        )
                )

        self.up = nn.ModuleList([])

        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [
                    ResBlock(
                            ch + inp_chs,
                            time_embed_dim=time_embed_dim,
                            out_channels=base_channels * mult,
                            dropout=dropout,
                            control = control,
                            img_size=cur_size
                            )
                    ]
                ch = base_channels * mult
                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels
                                    ),
                            )

                if i and j == num_res_blocks:
                    out_channels = ch

                    cur_size *= 2
                    layers.append(
                            ResBlock(
                                    ch,
                                    time_embed_dim=time_embed_dim,
                                    out_channels=out_channels,
                                    dropout=dropout,
                                    up=True,
                                    control = control,
                                    img_size=cur_size

                            )
                            if biggan_updown
                            else
                            Upsample(ch, conv_resample, out_channels=out_channels)
                            )
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))

        self.out1 = nn.Sequential(
                GroupNorm32(32, ch),
                nn.SiLU(),
                zero_module(nn.Conv2d(base_channels * channel_mults[0], self.out_channels, 3, padding=1))
                )


    def forward(self, x, time,control):


        time_embed = self.time_embedding(time)

        skips = []

        h = x.type(self.dtype)
        for i, m_base in enumerate(self.down):
            h = m_base(h, time_embed,control)
            skips.append(h)

        features = self.residual_blocks(h, time_embed,control).view(h.shape[:2])
        classes = self.cls_head(features)
        control["features"]=features
        control["classes"]=classes

        h = self.middle(h, time_embed,control)

        for i, module in enumerate(self.up):
            h_ctrl =  skips.pop()
            h = torch.cat([h, h_ctrl], dim=1)
            h = module(h, time_embed,control)
        h = h.type(x.dtype)
        h = self.out1(h)

        return h,classes,control


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # for p in module.parameters():
    #     p.detach().zero_()
    return module

if __name__ == "__main__":
    args = {
        'img_size':          256,
        'base_channels':     64,
        'dropout':           0.3,
        'num_heads':         4,
        'num_head_channels': '32,16,8',
        'lr':                1e-4,
        'Batch_Size':        64
        }
    model = UNetModel(
            args['img_size'], args['base_channels'], dropout=args[
                "dropout"], n_heads=args["num_heads"], attention_resolutions=args["num_head_channels"],
            in_channels=3
            )

    x = torch.randn(1, 3, 512, 512)
    t_batch = torch.tensor([1], device=x.device).repeat(x.shape[0])
    print(model(x, t_batch).shape)