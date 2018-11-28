import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable


def for_hook(module, input, output):
    # print(module)
    for val in input:
        print("input val:",val.shape)
    for out_val in output:
        print("output val:", out_val.shape)

model = models.vgg16()
layers = list(model.children())
# print (layers)
# model.register_forward_hook(for_hook)
layers[0][0].register_forward_hook(for_hook)
w = 224
# print ('layers: ', layers)
# target_layer = layers[0][0]
# print ('target layer:', target_layer)
input = np.ones(w*w*3).reshape([1,3,w,w])
input = torch.from_numpy(input)
input = input.type(torch.FloatTensor)
# print (input)

upsample = nn.Upsample(scale_factor=2, mode='bilinear')
a = upsample(input)
print (a.shape)


raise
output = model(input)


output_target = target_layer(input)
all = model.children()(input)
print ('all: ', all)
print('output shape:', output.shape)
print('output_target shape:', output_target.shape)
# print ('scuceed')

# v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
# h = v.register_hook(lambda grad: grad * 20)  # double the gradient
# v.backward(torch.Tensor([1, 1, 1]))
# print(v.grad.data)
# h.remove()  # removes the hook
#
# v.grad.data.zero_()
# v.backward(torch.Tensor([1, 1, 1]))
# print(v.grad.data)
