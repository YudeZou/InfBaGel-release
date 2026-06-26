import torch
import clip


def get_clip_features(raw_text: str):
    '''
    return clip features for a given text prompt
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-L/14@336px', device=device)
    model.eval()
    text = clip.tokenize([raw_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    # normalize the features
    text_features /= torch.norm(text_features, dim=-1, keepdim=True)
    assert text_features.shape == (1, 768)

    return text_features.detach().float()
