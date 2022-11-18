# import os
# import torch.onnx
#
# os.chdir('yolov7')
# model = torch.hub.load('.', 'custom', 'C:\\Users\\maxfyk\\Desktop\\best 3.pt', source='local')
# x = torch.randn(1, 3, 640, 640, requires_grad=True)
#
# torch.onnx.export(model,  # model being run
#                   x,  # model input (or a tuple for multiple inputs)
#                   "C:\\Users\\maxfyk\\Documents\\coji\\coji-api-mvp\\api\\statics\\styles\\geom-original\model.onnx",
#                   # where to save the model (can be a file or file-like object)
#                   # opset_version=10,  # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],  # the model's input names
#                   output_names=['output'],  # the model's output names
#                   dynamic_axes=None,
#                   verbose=False, )
import os
import torch.onnx
from PIL import Image
import numpy as np

os.chdir('yolov7')
im1 = Image.open('C:\\Users\\maxfyk\\Desktop\\test_o.jpg')
model = torch.hub.load('.', 'custom', 'C:\\Users\\maxfyk\\Desktop\\best 3.pt', source='local')
results = model([im1])
prediction = results.pred[0].tolist()
[print(p) for p in prediction]
# results.show()