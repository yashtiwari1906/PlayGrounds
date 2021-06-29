import torch


def content_loss(cont_feat, gen_feat):
    l = 1/2 * torch.mean((cont_feat - gen_feat)**2)
    return l

def style_loss(style_feat, gen_feat):
    batch_size, channel, height, width = gen_feat.shape
    G = torch.mm(gen_feat.view(channel, (height*width)), gen_feat.view(channel, (height*width)).t())
    A = torch.mm(style_feat.view(channel, (height*width)), style_feat.view(channel, (height*width)).t())
    l = torch.mean((G - A)**2)
    return l


def total_loss(cont_feat, style_feat, gen_feat):
    alpha, beta = 8, 70
    content_loss_ = style_loss_ = 0
    for con, style, gen in zip(cont_feat, style_feat, gen_feat):
        content_loss_+=content_loss(con, gen)
        style_loss_ += style_loss(style, gen)
    loss = alpha * content_loss_+ beta * style_loss_
    return loss