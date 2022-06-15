from torch.hub import load_state_dict_from_url

URL = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'

load_state_dict_from_url(URL)