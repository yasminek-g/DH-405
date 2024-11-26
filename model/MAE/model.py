import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import torch.nn.functional as F

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                #  image_size=32, # replaced by dynamic patching without limitation to image size
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        # Placeholder for pos_embedding, which will be adjusted dynamically
        self.pos_embedding = None

        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        if self.pos_embedding is not None:
            trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        B, C, H, W = img.size()  # Get the dimensions of the input image (B, C, H, W)

        # Calculate padding to make H and W divisible by patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        img = F.pad(img, (0, pad_w, 0, pad_h), "constant", 0)  # Apply padding

        # Update H and W after padding
        H, W = img.shape[2], img.shape[3]

        # Patchify the image
        patches = self.patchify(img)  # Shape: (B, emb_dim, H // patch_size, W // patch_size)
        
        # Flatten patches to a sequence
        patches = rearrange(patches, 'b c h w -> (h w) b c')  # Shape: (num_patches, B, emb_dim)

        # Dynamically adjust position embeddings based on number of patches
        num_patches = patches.size(0)
        if self.pos_embedding is None or self.pos_embedding.size(0) != num_patches:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches, 1, self.emb_dim))
            trunc_normal_(self.pos_embedding, std=0.02)

        # Add positional embeddings to patches
        patches = patches + self.pos_embedding

        # Apply patch shuffling and obtain forward and backward indexes
        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        # Add class token to the sequence
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        
        # Rearrange for transformer input
        patches = rearrange(patches, 't b c -> b t c')

        # Pass through the transformer and apply layer normalization
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                #  image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        # Placeholder for dynamic positional embedding
        self.pos_embedding = None

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        # Head for patch reconstruction
        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        if self.pos_embedding is not None:
            trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes, img_height, img_width):
        T = features.shape[0]
        
        # Adjust backward_indexes for class token and increment positions
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        
        # Add mask token to features to fill masked patches
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        
        # Reorder features to match the original unshuffled sequence
        features = take_indexes(features, backward_indexes)

        # Dynamic positional embedding based on the number of patches in this specific image
        num_patches = features.size(0)
        if self.pos_embedding is None or self.pos_embedding.size(0) != num_patches:
            self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches, 1, self.emb_dim))
            trunc_normal_(self.pos_embedding, std=0.02)
        
        # Add positional embedding to features
        features = features + self.pos_embedding

        # Transformer processing
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # Remove class token

        # Reconstruct patches
        patches = self.head(features)
        
        # Create mask for reconstructing the original image layout
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        
        # Adjust patch reshaping for the original image dimensions (after padding in encoder)
        H_patches = (img_height + self.patch_size - 1) // self.patch_size
        W_patches = (img_width + self.patch_size - 1) // self.patch_size
        patches = rearrange(patches, '(h w) b (c p1 p2) -> b c (h p1) (w p2)',
                            p1=self.patch_size, p2=self.patch_size, h=H_patches, w=W_patches)

        # Convert the mask patches back to the image format
        mask = rearrange(mask, '(h w) b (c p1 p2) -> b c (h p1) (w p2)',
                         p1=self.patch_size, p2=self.patch_size, h=H_patches, w=W_patches)

        return patches, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                #  image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        # Save original image dimensions
        img_height, img_width = img.shape[2], img.shape[3]

        # Pass the image through the encoder
        features, backward_indexes = self.encoder(img)

        # Pass the encoded features and original image dimensions to the decoder
        predicted_img, mask = self.decoder(features, backward_indexes, img_height, img_width)

        return predicted_img, mask

# class ViT_Classifier(torch.nn.Module):
#     def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
#         super().__init__()
#         self.cls_token = encoder.cls_token
#         self.pos_embedding = encoder.pos_embedding
#         self.patchify = encoder.patchify
#         self.transformer = encoder.transformer
#         self.layer_norm = encoder.layer_norm
#         self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

#     def forward(self, img):
#         patches = self.patchify(img)
#         patches = rearrange(patches, 'b c h w -> (h w) b c')
#         patches = patches + self.pos_embedding
#         patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
#         patches = rearrange(patches, 't b c -> b t c')
#         features = self.layer_norm(self.transformer(patches))
#         features = rearrange(features, 'b t c -> t b c')
#         logits = self.head(features[0])
#         return logits


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
