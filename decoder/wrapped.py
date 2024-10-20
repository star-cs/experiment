import torch
from torch import nn

from typing import List, Tuple, Type

from sam2.modeling.sam2_utils import MLP, LayerNorm2d

class SAMImageEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        
        super(SAMImageEncodeWrapper, self).__init__()
        
        # 提取出来 Sam 的模块
        self.sam_img_encoder = ori_sam.image_encoder
        
        self.patch_size = self.sam_img_encoder.patch_size
        self.depth = self.sam_img_encoder.depth
        self.prompt_dim = self.sam_img_encoder.embed_dim
        self.embed_dim = self.sam_img_encoder.embed_dim
        self.img_size = self.sam_img_encoder.img_size
        self.global_index = self.sam_img_encoder.global_index
        
        if fix:
            for name, param in self.sam_img_encoder.named_parameters():
                param.requires_grad = False


    def forward(self, x, prompt_tokens: torch.Tensor = None):
        
        # prompt_tokens [b, depth, num_prompts, prompt_dim]
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
            
        for idx, blk in enumerate(self.sam_img_encoder.blocks):
            current_prompt = prompt_tokens[:, idx, :, :] if prompt_tokens is not None else None
            # current_prompt [b, num_prompts, prompt_dim]
            x = blk(x, prompt_tokens=current_prompt)
            
        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x
    
    
    def forward_patch_embed(self, x):
        x = self.sam_img_encoder.patch_embed(x)  #  ImageEncoderViT 里的 patch_embed
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
        return x
    
    

    def forward_block(self, x, idx):
        x = self.sam_img_encoder.blocks[idx](x)
        return x
    
    
    def forward_neck(self, x):
        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x


class SAMPromptEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMPromptEncodeWrapper, self).__init__()
        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            for name, param in self.sam_prompt_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()
    

class SAMMaskDecoderWrapper_Med(nn.Module):

    def __init__(self, 
                 ori_sam, 
                 transformer_dim: int = 256, 
                 num_multimask_outputs = 3, 
                 activation: Type[nn.Module] = nn.GELU,
                 iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256,
                 ):
        super(SAMMaskDecoderWrapper_Med, self).__init__()
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = ori_sam.sam_mask_decoder.transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        # multimask_output: bool,
        multi_scale_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multi_scale_feature = multi_scale_feature,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,  # 图像嵌入张量
        image_pe: torch.Tensor,  # 图像位置编码张量
        sparse_prompt_embeddings: torch.Tensor,  # 稀疏提示嵌入张量
        dense_prompt_embeddings: torch.Tensor,  # 密集提示嵌入张量
        multi_scale_feature: torch.Tensor  # 多尺度特征张量
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据给定的图像和提示嵌入生成掩码预测。

        参数:
        - image_embeddings: 图像嵌入张量。
        - image_pe: 图像位置编码张量。
        - sparse_prompt_embeddings: 稀疏提示嵌入张量。
        - dense_prompt_embeddings: 密集提示嵌入张量。
        - multi_scale_feature: 多尺度特征张量。

        返回:
        - masks_hq: 高质量的掩码预测张量。
        """
        
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        src = self.output_upscaling(src)
        # multi_scale_feature = multi_scale_feature.repeat(b, 1, 1, 1)  
        upscaled_embedding = self.embedding_maskfeature(src) + multi_scale_feature  # [2, 32, 64, 64]
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
