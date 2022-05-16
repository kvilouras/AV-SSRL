import torch
import torch.nn as nn


__all__ = [
    'av_wrapper',
    'av_wrapper_sup'
]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dims=128):
        super(ProjectionHead, self).__init__()
        if not isinstance(proj_dims, list):
            proj_dims = [proj_dims]

        proj = []
        for d in proj_dims:
            proj.extend([nn.Linear(input_dim, d), nn.ReLU(inplace=True)])
            input_dim = d
        proj.pop(-1)
        self.projection = nn.Sequential(*proj)
        self.out_dim = proj_dims[-1]

    def forward(self, x):
        return self.projection(x)


class AVWrapper(nn.Module):
    def __init__(self, video_model, audio_model, proj_dim=128):
        super(AVWrapper, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.proj_dim = proj_dim

        if proj_dim:
            self.video_proj = ProjectionHead(video_model.out_dim, proj_dim)
            self.audio_proj = ProjectionHead(audio_model.out_dim, proj_dim)
            self.out_dim = self.video_proj.out_dim
        else:
            self.out_dim = video_model.out_dim

    def forward(self, video, audio):
        video_emb = self.video_model(video).squeeze()
        if self.proj_dim:
            video_emb = self.video_proj(video_emb)

        audio_emb = self.audio_model(audio).squeeze()
        if self.proj_dim:
            audio_emb = self.audio_proj(audio_emb)

        return video_emb, audio_emb


class AVWrapperSup(nn.Module):
    def __init__(self, video_model, audio_model, num_classes):
        super(AVWrapperSup, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.proj = nn.Linear(video_model.out_dim * 2, num_classes)  # cls head
        self.out_dim = video_model.out_dim

    def forward(self, video, audio):
        video_emb = self.video_model(video).squeeze()
        audio_emb = self.audio_model(audio).squeeze()
        # late fusion
        if video_emb.dim() == 1:
            out = torch.cat((video_emb, audio_emb), dim=0).unsqueeze(0)  # in case batch_size = 1
        else:
            out = torch.cat((video_emb, audio_emb), dim=1)

        return self.proj(out)


def av_wrapper(video_backbone, video_backbone_args, audio_backbone, audio_backbone_args, proj_dim=128, checkpoint=None):
    import models
    assert video_backbone in models.__dict__, 'Unknown model architecture'
    assert audio_backbone in models.__dict__, 'Unknown model architecture'
    video_model = models.__dict__[video_backbone](**video_backbone_args)
    audio_model = models.__dict__[audio_backbone](**audio_backbone_args)

    model = AVWrapper(video_model, audio_model, proj_dim=proj_dim)
    if checkpoint is not None:
        ckp = torch.load(checkpoint, map_location='cpu')
        nn.DataParallel(model).load_state_dict(ckp['model'])

    return model


def av_wrapper_sup(video_backbone, video_backbone_args, audio_backbone, audio_backbone_args, num_classes):
    import models
    assert video_backbone in models.__dict__, 'Unknown model architecture'
    assert audio_backbone in models.__dict__, 'Unknown model architecture'
    video_model = models.__dict__[video_backbone](**video_backbone_args)
    audio_model = models.__dict__[audio_backbone](**audio_backbone_args)

    model = AVWrapperSup(video_model, audio_model, num_classes)

    return model

