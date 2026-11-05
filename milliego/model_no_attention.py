"""Standalone radar reconstruction: official radar CNN + recurrent/head layers, no IMU/attention.
The public upstream repository provides only the fusion builder; this is not an author-supplied standalone checkpoint.
"""
from .model import RadarOnlyMilliEgo
class RadarOnlyNoAttention(RadarOnlyMilliEgo):
    def __init__(self):
        # Preserve all shared initial tensors for paired attention/no-attention comparisons.
        super().__init__()
        del self.mask
    def forward_features(self,z):
        b,t,c,h,w=z.shape
        z=z.permute(0,1,3,4,2).reshape(b,t,4096)
        z=self.lstm2(self.lstm1(z))
        import torch
        return torch.cat((self.translation(z),self.rotation(z)),-1)
