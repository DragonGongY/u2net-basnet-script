import os
from skimage import io, transform
import torch

import numpy as np
from PIL import Image

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():
    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp

    model_dir = "/media/dp/DATA/huihua_robot/U-2-Net-master/u2net_bce_itr_161000_train_0.623368_tar_0.060721.pth"

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    net.eval()

    input = torch.randn(1,3,320,320)

    inputs_test = input.cuda()

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    script_module = torch.jit.script(net)
    print(script_module.code)
    torch.jit.save(script_module, "script.pt")

    del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
