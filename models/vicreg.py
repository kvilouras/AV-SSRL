import torch
import torch.nn as nn

__all__ = [
    'vicreg_wrapper',
    'vicreg_v_wrapper'
]


class Expander(nn.Module):
    def __init__(self, input_dim, exp_dims=2048):
        super(Expander, self).__init__()
        if not isinstance(exp_dims, list):
            exp_dims = [exp_dims]

        expand = []
        for ex in exp_dims:
            expand.extend([nn.Linear(input_dim, ex), nn.BatchNorm1d(ex), nn.ReLU(inplace=True)])
            input_dim = ex
        expand = expand[:-2]
        self.expansion = nn.Sequential(*expand)
        self.out_dim = exp_dims[-1]

    def forward(self, x):
        return self.expansion(x)


class VICReg(nn.Module):
    def __init__(self, video_model, audio_model, exp_dims=2048, video_only=False):
        super(VICReg, self).__init__()
        self.video_only = video_only
        self.video_model = video_model
        if not video_only:
            self.audio_model = audio_model
        self.exp_dims = exp_dims

        if exp_dims:
            self.video_exp = Expander(video_model.out_dim, exp_dims)
            if not video_only:
                self.audio_exp = Expander(audio_model.out_dim, exp_dims)
            self.out_dim = self.video_exp.out_dim
        else:
            self.out_dim = video_model.out_dim

    def forward(self, video, audio):
        video_emb = self.video_model(video).squeeze()
        if self.exp_dims:
            video_emb = self.video_exp(video_emb)

        if self.video_only:
            # forward pass of the 2nd clip through the video model (named here as audio for variable re-usage)
            audio_emb = self.video_model(audio).squeeze()
            if self.exp_dims:
                audio_emb = self.video_exp(audio_emb)
        else:
            audio_emb = self.audio_model(audio).squeeze()
            if self.exp_dims:
                audio_emb = self.audio_exp(audio_emb)

        return video_emb, audio_emb


def vicreg_wrapper(video_backbone, video_backbone_args, audio_backbone, audio_backbone_args, exp_dim=2048,
                   checkpoint=None):
    import models
    assert video_backbone in models.__dict__, "Unknown model architecture"
    assert audio_backbone in models.__dict__, "Unknown model architecture"
    video_model = models.__dict__[video_backbone](**video_backbone_args)
    audio_model = models.__dict__[audio_backbone](**audio_backbone_args)

    model = VICReg(video_model, audio_model, exp_dim)
    if checkpoint is not None:
        ckp = torch.load(checkpoint, map_location='cpu')
        nn.DataParallel(model).load_state_dict(ckp['model'])

    return model


def vicreg_v_wrapper(video_backbone, video_backbone_args, exp_dim=2048, checkpoint=None):
    import models
    assert video_backbone in models.__dict__, "Unknown model architecture"
    video_model = models.__dict__[video_backbone](**video_backbone_args)

    model = VICReg(video_model, None, exp_dim, video_only=True)
    if checkpoint is not None:
        ckp = torch.load(checkpoint, map_location='cpu')
        nn.DataParallel(model).load_state_dict(ckp['model'])

    return model

