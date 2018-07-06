from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
from torch import optim
import random as rand
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tvt
import math
from imsitu import imSituVerbRoleNounEncoder 
from imsitu import imSituTensorEvaluation 
from imsitu import imSituTensorV2Evaluation 
from imsitu import imSituSituation 
from imsitu import imSituSimpleImageFolder
from utils import initLinear
from utils import initEmbeddings

import json

class vgg_modified(nn.Module):
  def __init__(self):
    super(vgg_modified,self).__init__()
    self.vgg = tv.models.vgg16(pretrained=True)
    self.features = self.vgg.features
    self.classifier = self.vgg.classifier
 
  def rep_size(self): return 4096

  def forward(self,x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    for i in range(0,len(self.classifier)-1):
       x = self.classifier[i](x) 
    return x
    #return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))

class resnet_modified_large(nn.Module):
 def __init__(self):
    super(resnet_modified_large, self).__init__()
    self.resnet = tv.models.resnet101(pretrained=True)
    #probably want linear, relu, dropout
    self.linear = nn.Linear(7*7*2048, 1024)
    self.dropout2d = nn.Dropout2d(.5)
    self.dropout = nn.Dropout(.5)
    self.relu = nn.LeakyReLU()
    initLinear(self.linear)

 def base_size(self): return 2048
 def rep_size(self): return 1024

 def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
     
        x = self.dropout2d(x)

        #print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))

class resnet_modified_medium(nn.Module):
 def __init__(self):
    super(resnet_modified_medium, self).__init__()
    self.resnet = tv.models.resnet50(pretrained=True)
    #probably want linear, relu, dropout
    #self.linear = nn.Linear(7*7*2048, 1024)
    #self.dropout2d = nn.Dropout2d(.5)
    self.dropout = nn.Dropout(global_dropout)
    self.relu = nn.LeakyReLU()
    #initLinear(self.linear)
    self.avg = nn.AvgPool2d(16)

 def base_size(self): return 2048
 def rep_size(self): return 2048

 def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
     
        #x = self.dropout2d(x)
        x = self.dropout(self.avg(x).view(-1, 2048))
        return x 
 

class resnet_modified_medium_spatial(nn.Module):
 def __init__(self):
    super(resnet_modified_medium_spatial, self).__init__()
    self.resnet = tv.models.resnet50(pretrained=True)
    self.l4 = []
    self.resnet2_l4_r = []
    
    #for i in range(0,6):
    #  resnet2_l4 = tv.models.resnet50(pretrained=True).layer4
    #  self.l4.append(resnet2_l4[0])
    #  self.resnet2_l4_r.append(nn.Sequential(resnet2_l4[1:]))
    #self.l4 = nn.ModuleList(self.l4)
    #self.resnet2_l4_r = nn.ModuleList(self.resnet2_l4_r)
    
    #self.dropout = nn.Dropout(global_dropout)

    self.avg = nn.AvgPool2d(16)
    self.combine = nn.Conv2d(512, 512, 1)
 
    inplanes = 2048
    planes = 1024
    self.combine_conv1 = nn.Conv2d(64+256+inplanes, planes, kernel_size=1, bias=False)
    self.combine_bn1 = []
    self.combine_bn2 = [] 
    self.combine_bn3 = []
    for i in range(0,6):
      self.combine_bn1.append(nn.BatchNorm2d(planes))
      self.combine_bn2.append(nn.BatchNorm2d(planes))
      self.combine_bn3.append(nn.BatchNorm2d(planes * 2))
  
    self.combine_bn1 = nn.ModuleList(self.combine_bn1)
    self.combine_bn2 = nn.ModuleList(self.combine_bn2)
    self.combine_bn3 = nn.ModuleList(self.combine_bn3)

    self.combine_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
    self.combine_conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
    self.combine_relu = nn.ReLU(inplace=False)
 
 def non_combine_params(self):
    rv = []
    rv.extend(self.resnet.parameters())
    #rv.extend(self.l4.parameters())
    #rv.extend(self.resnet2_l4_r.parameters())
    return rv
 
 def base_size(self): return 2048
 def rep_size(self): return 2048

 def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
#        cached = x
        #print(cached.size())
        #regulate here
        #x = x * y #a pointwise multiply to change what features are important
        x = self.resnet.layer4(x)
     
        #B x C x H x W
        #x = self.dropout2d(x)
        x2 = self.avg(x).view(-1, 2048)
        return (x, x2)
 
 def forward_regulated(self, i, x, residual):#, r, b):
        #residual = x
        bn1 = self.combine_bn1[i]
        bn2 = self.combine_bn2[i]
        bn3 = self.combine_bn3[i]

        out = self.combine_conv1(x) 
        out = bn1(out)
        #out = out*r + b
        out = self.combine_relu(out)

        out = self.combine_conv2(out)
        out = bn2(out)
        out = self.combine_relu(out)

        out = self.combine_conv3(out)
        out = bn3(out)

        out += residual
        out = self.combine_relu(out)
      
        x2 = self.avg(out).view(-1, 2048)

        return (out, x2)
         
        #return x#self.avg(x).view(-1, 2048)

class resnet_modified_medium_spatial_l4(nn.Module):
 def __init__(self):
    super(resnet_modified_medium_spatial_l4, self).__init__()
    resnet = tv.models.resnet50(pretrained=True)
    self.layer4 = resnet.layer4
    #self.resnet2 = tv.models.resnet50(pretrained=True)
    self.avg = nn.AvgPool2d(16)

 def base_size(self): return 2048
 def rep_size(self): return 2048

 def forward(self, x):
        x = self.layer4(x)
        #B x C x H x W
        x2 = self.avg(x).view(-1, 2048)
        return x2
 
class resnet_modified_small(nn.Module):
 def __init__(self):
    super(resnet_modified_small, self).__init__()
    self.resnet = tv.models.resnet34(pretrained=True)
    #probably want linear, relu, dropot
    self.linear = nn.Linear(7*7*512, 512)
    self.dropout2d = nn.Dropout2d(.5)
    self.dropout = nn.Dropout(global_dropout)
    self.relu = nn.LeakyReLU()
    self.bnl = nn.BatchNorm1d(1024)
    #initLinear(self.linear)
    self.avg = nn.AvgPool2d(16)

 def base_size(self): return 512
 def rep_size(self): return 512

 def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
     
        #x = self.dropout2d(x)
       
        #y = self.dropout(self.linear(x.view(-1, 7*7*self.base_size())))
        x = self.dropout(self.avg(x).view(-1,512))
        #print(x.size())
        return x 
        #return self.linear(x.view(-1, 7*7*self.base_size()))

class baseline_predictor(nn.Module):
   
   from retina.caffe2.retinanet import Caffe2Model
   from retina.caffe2.retinanet import get_targets
   from math import log
   def initConvs(self, modules, std=.01, b=0):
      for m in modules: 
        if isinstance(m, nn.Conv2d): initConv(m, std, b)
   
   def initConv(self, module, std=.01, b=0):
      module.weight.normal_(0, std)
      module.bias = module.bias.zero_()+b
 

   def __init__(self, encoding, pretrained, v_rep_size = 256, r_rep_size = 256, stack_depth = 4 ):

      self.stack = Caffe2Model("resnet50.protodump", "resnet50_init.paramdump", "gpu_0/data", "retnet")
      self.maps = ["gpu0/fpn7", "gpu0/fpn6", "gpu0/fpn_res3_3_sum", "gpu0/fpn_res4_5_sum", "gpu0/fpn_res5_2_sum"] #these are our input maps
      
      self.embedding_v = nn.Embedding(self.encoding.n_verbs(), v_rep_size) 
      self.embedding_r = nn.Embedding(self.encoding.n_roles(), r_rep_size)

      self.nouns = self.encoding.n_nouns()
     
      self.anchors = 9
      self.hd = 256
      self.b_init = -log((1.-.01)/.01)

      #init a few conv layers 
      classification_branch = []
      for i in range(stack_depth):
        prediction_branch.append(nn.Conv2d(self.hd, self.hd, 3))
        prediction_branch.append(nn.ReLU())

      self.initConvs(classification_branch)    
      self.classification_branch = nn.ModuleList(classification_branch)

      self.classification = nn.Conv2d(self.hd, self.anchors*self.nouns)
      self.initConv(self.classification, std=.01, b=self.b_init)

      #init a few conv layers 
      regression_branch = []
      for i in range(stack_depth):
        regression_branch.append(nn.Conv2d(self.hd, self.hd, 3))
        regression_branch.append(nn.ReLU())

      self.initConvs(regression_branch)
      self.regression_branch = nn.ModuleList(regression_branch)     
 
      self.regression = nn.Conv2d(self.hd, self.anchors*4)
      self.initConv(self.regression)      
 
      #init the combination layer
      self.combination_prediction = nn.Conv2d(v_rep_size + r_rep_size + self.hd, self.hd, 3)
      self.combination_regression = nn.Conv2d(v_rep_size + r_rep_size + self.hd, self.hd, 3)
      self.initConvs([self.combination_prediction, self.combination_regression])


  #assumes exactly one instance
   def prepareLabels(self, labels, boxes, im_info):
      blobs = []
      for i in range(6): blobs.append(get_targets(im_info, [boxes[i]], im_info))
      return blobs

   def apply(self, branch, final, m):
      for module in branch: m = module(m)
      return final(m)

   def detection(self, tensors, roles):
      for role in range(6):
        for m in self.maps:
          
          initial_map = tensors[m]
          
          current_map = self.combine(combination_prediction, initial_map, roles)  

          o = self.apply(classification_branch, classification, current_map)
          tensors[m+"_classification_"+role] = o

          current_map = self.combine(combination_regression, initial_map, roles)  

          o = self.apply(regression_branch, regression, current_map)
          tensors[m+"_regression_"+role] = o
      return outputs

   def forward(self, vals, keys):
      targets = {}
      for i in range(0,len(vals)): 
          _v = vals[i]
          if len(vals[i].size()) > 1 and vals[i].size()[1] == 1: _v = _v.squeeze(1)
          targets[keys[i]] = _v 
      image = targets["im_data"]
      
      outputs = self.stack(image)
      outputs = self.detection(outputs)
#      if "gt_boxes" in keys:

      #compute losses
  
class rnn_predictor_spatial(nn.Module):
   def train_preprocess(self): return self.train_transform
   def dev_preprocess(self): return self.dev_transform
   def set_train_mode(self, mode): self.train_mode = mode
   #these seem like decent splits of imsitu, freq = 0,50,100,282 , prediction type can be "max_max" or "max_marginal"
   def __init__(self, encoding, ngpus = 4, cnn_type = "resnet_34", v_rep_size=256, r_rep_size=256, n_rep_size = 256, hidden_size = 1024, norm_vision = False, feed_self = False, fine_tune_cnn = True, attention = False ):
     super(rnn_predictor_spatial, self).__init__()
     self.attention = attention 
     self.hidden_size = hidden_size
     self.norm_vision = norm_vision
     self.feed_self = feed_self
     self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     self.train_transform = tv.transforms.Compose([
            #tv.transforms.Scale(512),
            tv.transforms.RandomCrop(512),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])
     self.dev_transform = tv.transforms.Compose([
            #tv.transforms.Scale(512),
            tv.transforms.CenterCrop(512),
            tv.transforms.ToTensor(),
            self.normalize,
        ])


     self.n_verbs = encoding.n_verbs()
     self.loss_n = nn.CrossEntropyLoss(ignore_index = encoding.pad_symbol())
     self.loss_v = nn.CrossEntropyLoss()
     
     self.encoding = encoding
     
     #cnn
     print(cnn_type)
     if cnn_type == "resnet_101" : 
     #  self.cnn_v = resnet_modified_large()
       self.cnn_n = resnet_modified_large()
     elif cnn_type == "resnet_50": 
    #   self.cnn_v = resnet_modified_medium()
       self.cnn_n = resnet_modified_medium_spatial()
       #l4 = []
       #for i in range(0,6):
       # l4.append(resnet_modified_medium_spatial_l4())
       #self.cnn_l4 = nn.ModuleList(l4)
     elif cnn_type == "resnet_34": 
    #   self.cnn_v = resnet_modified_small()
       self.cnn_n= resnet_modified_small()
     elif cnn_type == "vgg":
       #self.cnn_v = vgg_modified()
       self.cnn_n = vgg_modified()
     else:
       print("unknown base network" )
       exit()
     
     self.rep_size = self.cnn_n.rep_size()
     self.embedding_v = nn.Embedding(self.encoding.n_verbs(), v_rep_size) 
     self.embedding_n = nn.Embedding(self.encoding.n_nouns()+1, n_rep_size)
     self.embedding_r = nn.Embedding(self.encoding.n_roles(), r_rep_size)
     self.noun_model = nn.LSTM(self.rep_size+n_rep_size+r_rep_size+v_rep_size,hidden_size,1, batch_first = True)
     self.verb_linear = nn.Linear(self.rep_size, self.encoding.n_verbs())
     self.noun_project = nn.Linear(self.rep_size, n_rep_size)     
     self.noun_linear = nn.Linear(hidden_size, self.encoding.n_nouns())
     
     self.attend_rep = 32
     self.attention_network_c1 = nn.Conv2d(v_rep_size + r_rep_size + self.rep_size, self.rep_size, 1)
     self.attention_network_relu1 = nn.ReLU()
     self.attention_network_bn1 = nn.BatchNorm2d(self.rep_size) 
     self.attention_network_c2 = nn.Conv2d(self.rep_size+1, self.attend_rep, 3, padding = 1)
     self.attention_network_relu2 = nn.ReLU()
     self.attention_network_c3 = nn.Conv2d(self.attend_rep, self.attend_rep, 3, padding = 1)
     self.attention_network_relu3 = nn.ReLU()
     self.attention_network_bn2 = nn.BatchNorm2d(self.attend_rep)
     self.attention_network_c4 = nn.Conv2d(self.attend_rep, 1, 1)   
     self.attention_avg = nn.AvgPool2d(16)
    
     self.regulator_output = 1024
     self.regulator_l1 = nn.Linear(v_rep_size, self.regulator_output)
     self.regulator_l2 = nn.Linear(v_rep_size, self.regulator_output)

     self.regulator_nl = torch.tanh 

     self.verb_list = torch.autograd.Variable(torch.LongTensor(range(0,self.encoding.n_verbs())).cuda())
     
     self.role_vectors = []
     allroles = []
     for i in range(self.encoding.n_verbs()):
       roleids = []
       role_index = self.encoding.v_order[i]
       order = sorted(role_index.items(), key=lambda x: x[1]) 
       for _o in order: roleids.append(_o[0])
       while len(roleids) < self.encoding.mr: roleids.append(self.encoding.pad_symbol())
       allroles.extend(roleids)
       self.role_vectors.append(torch.LongTensor(roleids))
     self.role_list = torch.autograd.Variable(torch.LongTensor(allroles).cuda()).view(self.encoding.n_verbs(),-1)
 
     self.dropout = nn.Dropout(global_dropout)

     #initLinear(self.attention_network_l1)
     #initLinear(self.attention_network_l2) 
     initLinear(self.noun_project) 
     initLinear(self.verb_linear)
     initLinear(self.noun_linear) 
     initLinear(self.regulator_l1)
     initLinear(self.regulator_l2)
     initEmbeddings(self.embedding_v, word_val = 1)
     initEmbeddings(self.embedding_n, word_val = 1)
     initEmbeddings(self.embedding_r, word_val = 1)
  
     for p in self.noun_model.parameters():
       if len(p.size()) < 2: continue
       #print(p)
       nn.init.orthogonal(p.data)
       #print(p) 
 
     if fine_tune_cnn == 0: self.set_ft(False)
     else: self.set_ft(True)

     self.noun_model.flatten_parameters()

   def set_ft(self, v):
     print("setting ft to {}".format(v))
     for i in self.cnn_n.non_combine_params(): i.requires_grad = v
     #for i in self.cnn_l4.parameters(): i.requires_grad = v

   def get_roles(self,verb_vector):
     rv = []
     for _v in verb_vector.cpu().view(-1):
       rv.append(self.role_vectors[_v].view(1,-1))
     return torch.cat(rv,1)

   def opt_parameters(self):
     rv = []
     for i in self.parameters():
       if i.requires_grad: rv.append(i)
     return rv

   def attention_network(self, base, x, prev_a):
     b ,c ,  h, w  = x.size()
     #x = x.view(b*h*w, c)
     _x = self.attention_network_c1(x) #this would probably be better with a 1x1 conv
     _x = self.attention_network_relu1(_x)
     _x = self.attention_network_bn1(_x)
     x2 = _x + base
     
     #x = x.view(b , h , w , self.attend_rep).permute(0, 3, 1, 2)
     #print(x.size())
     #print(prev_a.size())
     x = torch.cat([x2,prev_a.view(b, 1 , h, w)], 1)
     x = self.attention_network_c2(x)
     x = self.attention_network_relu2(x)
     x = self.attention_network_c3(x)
     x = self.attention_network_relu3(x)
     x = self.attention_network_bn2(x)
     #x = x.permute(0, 2, 3, 1).contiguous().view(b*h*w, self.attend_rep)
     #b , 1 , h , w 
     x = self.attention_network_c4(x).view(b, h , w)

     return (x, x2.permute(0, 2, 3, 1))

   def image_rep(self, spatial_rep, hidden_state, prev_attention):
     if self.attention == False:
       return (self.attention_avg(spatial_rep), None)
     
     #should return a single vector
     #make this batch x h x w x c
     b , vc , h , w = spatial_rep.size()
     #print(spatial_rep.size()) 
     #print(hidden_state.size())
     _,_ , hc = hidden_state.size()
     x = spatial_rep.permute(0, 2, 3, 1).view(b, h*w, vc)
     #print(x.size())
     #hidden state is batch x c
     y = hidden_state.view(b, 1, hc).expand(b, h*w, hc)
     #print(y.size())
     z = torch.cat([x , y], 2).view(b, h, w , hc + vc).permute(0, 3 , 2, 1)
     #print(z.size())
     (_p, x2) = self.attention_network(spatial_rep, z,  prev_attention)
     _p = _p.view(b, h*w)
     x2 = x2.view(b, h*w, -1) 
     #print(_p.size())
     p = nn.functional.softmax(_p, dim = 1)
     cp = p.view(b, h*w, 1).expand(b, h*w, x2.size()[2])
     #print(p.size())
     #self.pprint(p[0].view(h,w))
     #print("--")
     #self.pprint(p[0].view(h,w))
     #print("--")
     #print(x2.size()) 
     #print(cp.size())
     rep = torch.sum(cp*x2, 1)
     #rep = self.attention_avg(x2)
     #print(rep.size())
     return (rep, p)

   def pprint(self, t):
     x, y  = t.size()
     
     for i in range(0,x):
       os = ""
       for j  in range(0,y): os += "{:.5f}\t".format(t[i,j].item())
       print(os)
    
   def forward(self, image, control, target = None):
     #b , c, h , w = image.size()
     if control == "topk": return self.forward_test_topk(image, target) 
     elif control == "topk_verbs": return self.forward_test_topk_verb(image, target) 
    # elif control == "given_verb": return self.forward_test_given_verb(image, target)
      

     #we need a preterminal spatial map
     #spatial_rep_n, rep_n, cache = self.cnn_n(image)
     spatial_rep, rep_n = self.cnn_n(image)
     cache_b , cache_c, cache_h, cache_w = spatial_rep.size()
     cache_transposed = spatial_rep.permute(0, 2, 3, 1)
     rep_v = rep_n
     v_pot = self.verb_linear(rep_v)
     b = target.size()[0]
     verbs = target[:,0].contiguous().view(b).contiguous()

     ve = self.embedding_v(verbs)
     b, e = ve.size()   
     #viz = self.noun_project(rep_n).view(b, 1, -1)
     viz = rep_n
 
     mr = self.encoding.max_roles()
     nouns = []
     roles = []
     #make copies of each of nouns, make copies of the compute so far.
    # if self.train: 
     n = target[:,2::2].contiguous().view(b,3,mr)
     r = target[:,1::2].contiguous().view(b,3,mr)
     for i in range(0,mr): nouns.append( n[:,int(3*rand.random()), i] )
     for i in range(0,mr): roles.append(r[:,0,i])

     noun_pots = []
     noun_maxes = []
     mi = Variable(torch.LongTensor([self.encoding.n_nouns()]).cuda()).expand(b) #start symbol
     statep = (Variable(torch.zeros((1, b, self.hidden_size)).cuda()), None)
          #prev_attn = Variable(torch.zeros((b, 1, 16, 16)).cuda()) #hard coded for 512 :/
     #viz = rep_n

          #regulator = self.regulator_l1(regulator_input).view(cache_b, 1, cache_c/2).expand(cache_b, cache_h*cache_w, cache_c/2).view(cache_b, cache_h, cache_w, cache_c/2)
     #regulator2 = self.regulator_l2(regulator_input).view(cache_b, 1, cache_c/2).expand(cache_b, cache_h*cache_w, cache_c/2).view(cache_b, cache_h, cache_w, cache_c/2)
     #regulator = regulator.permute(0,3,1,2)
     #regulator2 = regulator2.permute(0,3,1,2)

     for x in range(0, mr):
      # (viz, prev_attn) = self.image_rep(spatial_rep_n, torch.cat([ve.view(b,1,-1), self.embedding_r(roles[x]).view(b,1,-1)], dim=2), prev_attn)
       regulator_input = torch.cat([torch.cat([ve.view(b,-1), self.embedding_r(roles[x]).view(b,-1)], dim=1).view(b,1,-1).expand(b, cache_h*cache_w, -1).view(b, cache_h, cache_w, -1), cache_transposed], 3).permute(0, 3, 1, 2)

       (_,viz) = self.cnn_n.forward_regulated(x, regulator_input, spatial_rep)#spatial_rep)#, regulator, regulator2)
       outputp = torch.cat([ve.view(b,1,-1), self.embedding_r(roles[x]).view(b,1,-1),self.embedding_n(mi).view(b, 1, -1),viz.view(b,1,-1)], dim=2)
     
       if x > 0 : output, statex = self.noun_model(outputp, statep)
       else: output, statex = self.noun_model(outputp, None)

       noun_pot = self.dropout(self.noun_linear(output.view(b,-1)))
       noun_pots.append(noun_pot)
       mv, mi = torch.max(noun_pot, dim = 1)
       noun_maxes.append(mi.view(b, 1))
       #if self.train: outputp = self.embedding_n(nouns[x])
       #else:
       #if feed_self is true, you DO NOT use gt 
       if control != "given_verb" and not self.feed_self and self.train: 
         mi = nouns[x]
       statep = statex
       #B x C
       #regulate the cnn with information 
      
     max_i = torch.cat(noun_maxes, dim=1).view(b, mr)
     
     return (v_pot, max_i, torch.cat(noun_pots, dim = 1).view(b, 1, mr, -1), v_pot) 

   def forward_test_given_verb(self, image, verbs):
     spatial_rep_n, rep_n, cache = self.cnn_n(image)

     cache_b , cache_c, cache_h, cache_w = cache.size()
     cache_transposed = cache.permute(0, 2, 3, 1)

     rep_v = rep_n
     v_pot = self.verb_linear(rep_v)
     
     b = image.size()[0]

     ve = self.embedding_v(verbs)
     b, e = ve.size()   
     #viz = self.noun_project(rep_n).view(b, 1, -1)
     viz = rep_n
 
     mr = self.encoding.max_roles()
     roles = Variable(self.get_roles(verbs).cuda()).view(b, mr)

     noun_pots = []
     noun_maxes = []
     mi = Variable(torch.LongTensor([self.encoding.n_nouns()]).cuda()).expand(b) #start symbol
     
     statep = (Variable(torch.zeros((1, b, self.hidden_size)).cuda()), None)
     prev_attn = Variable(torch.zeros((b, 1, 16, 16)).cuda()) #hard coded for 512 :/

     viz = rep_n
     for x in range(0, mr):
       #(viz, prev_attn) = self.image_rep(spatial_rep_n, torch.cat([ve.view(b,1,-1), self.embedding_r(roles[:,x]).view(b,1,-1)], dim=2), prev_attn)

       regulator_input = torch.cat([ve.view(b,-1), self.embedding_r(roles[:,x]).view(b,-1),self.embedding_n(mi).view(b, -1),statep[0].view(b, -1)], dim=1)
       regulator = self.regulator_l1(regulator_input).view(cache_b, 1, cache_c/2).expand(cache_b, cache_h*cache_w, cache_c/2).view(cache_b, cache_h, cache_w, cache_c/2)
       regulator2 = self.regulator_l2(regulator_input).view(cache_b, 1, cache_c/2).expand(cache_b, cache_h*cache_w, cache_c/2).view(cache_b, cache_h, cache_w, cache_c/2)
       regulator = regulator.permute(0,3,1,2)
       regulator2 = regulator2.permute(0,3,1,2)
       viz = self.attention_avg(self.cnn_n.forward_regulated(cache, regulator, regulator2))

       outputp = torch.cat([ve.view(b,1,-1), self.embedding_r(roles[:,x]).view(b,1,-1),self.embedding_n(mi).view(b, 1, -1),viz.view(b,1,-1)], dim=2)
     
       if x > 0 : output, statex = self.noun_model(outputp, statep)
       else: output, statex = self.noun_model(outputp, None)
       noun_pot = self.dropout(self.noun_linear(output.view(b,-1)))
       noun_pots.append(noun_pot)
       mv, mi = torch.max(noun_pot, dim = 1)
       noun_maxes.append(mi.view(b, 1))
       statep = statex



       #regulator = torch.tanh(self.regulator_l1(statep[0])).view(cache_b, 1, cache_c).expand(cache_b, cache_h*cache_w, cache_c).view(cache_b, cache_h, cache_w, cache_c)
       #new_image = regulator*cache_transposed
       #new_image = new_image.permute(0,3,1,2)
       #print(new_image.size())

       #viz = self.attention_avg(self.cnn_n.forward_regulated(new_image))
       #viz = self.cnn_l4[x](new_image)
       #viz = self.cnn_n.forward_regulated((regulator*cache_transposed).permute(0, 3, 1, 2))

     max_i = torch.cat(noun_maxes, dim=1).view(b, mr)
     return max_i

   def forward_test_topk_verb(self, image, k):
     #seperate reps
     spatial_rep_n, rep_n = self.cnn_n(image)
     v_pot = self.verb_linear(rep_n)
     b, _ = rep_n.size()
     verb_values, verb_indicies = torch.topk(v_pot, k, 1)
     return (verb_values.view(b, k), verb_indicies.view(b,k)) 

   def loss(self, v_pot, n_pots, target):
     (b, nv, mr, n) = n_pots.size() 
     verbs = target[:,0].contiguous().view(b).contiguous()
     v = self.loss_v(v_pot, verbs)
     #print(n_pots.size())
     #print(n_pots)
     n_pots = n_pots.expand(b, 3, mr, n).contiguous().view(b*3*mr, n)
     #print(n_pots[0:6, :])
     #print(n_pots[6:12, :])
     n = self.loss_n(n_pots, target[:,2::2].contiguous().view(b*3*mr))  
     #print (v+n) 
     return v + 6*n #do we want to scale the noun loss?
     
class rnn_predictor(nn.Module):
   def train_preprocess(self): return self.train_transform
   def dev_preprocess(self): return self.dev_transform
   def set_train_mode(self, mode): self.train_mode = mode
   #these seem like decent splits of imsitu, freq = 0,50,100,282 , prediction type can be "max_max" or "max_marginal"
   def __init__(self, encoding, ngpus = 4, cnn_type = "resnet_34", v_rep_size=256, r_rep_size=256, n_rep_size = 256, hidden_size = 1024, norm_vision = False, feed_self = False, fine_tune_cnn = True ):
     super(rnn_predictor, self).__init__() 
     self.hidden_size = hidden_size
     self.norm_vision = norm_vision
     self.feed_self = feed_self
     self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     self.train_transform = tv.transforms.Compose([
            #tv.transforms.Scale(512),
            tv.transforms.RandomCrop(512),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

     self.dev_transform = tv.transforms.Compose([
            #tv.transforms.Scale(256),
            tv.transforms.CenterCrop(512),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

     self.n_verbs = encoding.n_verbs()
     self.loss_n = nn.CrossEntropyLoss(ignore_index = encoding.pad_symbol())
     self.loss_v = nn.CrossEntropyLoss()
     
     self.encoding = encoding
     
     #cnn
     print(cnn_type)
     if cnn_type == "resnet_101" : 
     #  self.cnn_v = resnet_modified_large()
       self.cnn_n = resnet_modified_large()
     elif cnn_type == "resnet_50": 
    #   self.cnn_v = resnet_modified_medium()
       self.cnn_n = resnet_modified_medium()
     elif cnn_type == "resnet_34": 
    #   self.cnn_v = resnet_modified_small()
       self.cnn_n= resnet_modified_small()
     elif cnn_type == "vgg":
       #self.cnn_v = vgg_modified()
       self.cnn_n = vgg_modified()
     else:
       print("unknown base network" )
       exit()
     
     self.rep_size = self.cnn_n.rep_size()
     self.embedding_v = nn.Embedding(self.encoding.n_verbs(), v_rep_size) 
     self.embedding_n = nn.Embedding(self.encoding.n_nouns()+1, n_rep_size)
     self.embedding_r = nn.Embedding(self.encoding.n_roles(), r_rep_size)
     self.noun_model = nn.LSTM(self.rep_size+n_rep_size+r_rep_size+v_rep_size,hidden_size,1, batch_first = True)
     self.verb_linear = nn.Linear(self.rep_size, self.encoding.n_verbs())
     self.noun_project = nn.Linear(self.rep_size, n_rep_size)     
     self.noun_linear = nn.Linear(hidden_size, self.encoding.n_nouns())
     self.verb_list = torch.autograd.Variable(torch.LongTensor(range(0,self.encoding.n_verbs())).cuda())
     
     self.role_vectors = []
     allroles = []
     for i in range(self.encoding.n_verbs()):
       roleids = []
       role_index = self.encoding.v_order[i]
       order = sorted(role_index.items(), key=lambda x: x[1]) 
       for _o in order: roleids.append(_o[0])
       while len(roleids) < self.encoding.mr: roleids.append(self.encoding.pad_symbol())
       allroles.extend(roleids)
       self.role_vectors.append(torch.LongTensor(roleids))
     self.role_list = torch.autograd.Variable(torch.LongTensor(allroles).cuda()).view(self.encoding.n_verbs(),-1)
 
     self.dropout = nn.Dropout(global_dropout)
     
     initLinear(self.noun_project) 
     initLinear(self.verb_linear)
     initLinear(self.noun_linear) 
     initEmbeddings(self.embedding_v, word_val = 1)
     initEmbeddings(self.embedding_n, word_val = 1)
     initEmbeddings(self.embedding_r, word_val = 1)
  
     for p in self.noun_model.parameters():
       if len(p.size()) < 2: continue
       #print(p)
       nn.init.orthogonal(p.data)
       #print(p) 


     if not fine_tune_cnn: 
       for i in self.cnn_n.parameters(): i.requires_grad = False
 
   def get_roles(self,verb_vector):
     rv = []
     for _v in verb_vector.cpu().view(-1):
       rv.append(self.role_vectors[_v].view(1,-1))
     return torch.cat(rv,1)

   def opt_parameters(self):
     rv = []
     for i in self.parameters():
       if i.requires_grad: rv.append(i)
     return rv

   def forward(self, image, control, target = None):
     if control == "topk": return self.forward_test_topk(image, target) 
     elif control == "topk_verbs": return self.forward_test_topk_verb(image, target) 
     elif control == "given_verb": return self.forward_test_given_verb(image, target)
      
     rep_n = self.cnn_n(image)
     rep_v = rep_n
     v_pot = self.verb_linear(rep_v)
     
     b = target.size()[0]
     verbs = target[:,0].contiguous().view(b).contiguous()

     ve = self.embedding_v(verbs)
     b, e = ve.size()   
     viz = self.noun_project(rep_n).view(b, 1, -1)
     viz = rep_n
 
     mr = self.encoding.max_roles()
     nouns = []
     roles = []
     #make copies of each of nouns, make copies of the compute so far.
     if self.train: 
       n = target[:,2::2].contiguous().view(b,3,mr)
       r = target[:,1::2].contiguous().view(b,3,mr)
       for i in range(0,mr): nouns.append( n[:,int(3*rand.random()), i] )
       for i in range(0,mr): roles.append(r[:,0,i])

     noun_pots = []
     noun_maxes = []
     mi = Variable(torch.LongTensor([self.encoding.n_nouns()]).cuda()).expand(b) #start symbol
     statep = None
     for x in range(0, mr):
       outputp = torch.cat([ve.view(b,1,-1), self.embedding_r(roles[x]).view(b,1,-1),self.embedding_n(mi).view(b, 1, -1),viz.view(b,1,-1)], dim=2)
       #print(outputp.size())
       output, statex = self.noun_model(outputp, statep) 
       noun_pot = self.dropout(self.noun_linear(output.view(b,-1)))
       noun_pots.append(noun_pot)
       mv, mi = torch.max(noun_pot, dim = 1)
       noun_maxes.append(mi.view(b, 1))
       #if self.train: outputp = self.embedding_n(nouns[x])
       #else:
       #if feed_self is true, you DO NOT use gt 
       if not self.feed_self and self.train: 
         mi = nouns[x]
       statep = statex

     max_i = torch.cat(noun_maxes, dim=1).view(b, mr)
     
     return (v_pot, max_i, torch.cat(noun_pots, dim = 1).view(b, 1, mr, -1), v_pot) 

   def forward_test_given_verb(self, image, verbs):
     rep_n = self.cnn_n(image)
     rep_v = rep_n
     v_pot = self.verb_linear(rep_v)
     
     b = image.size()[0]

     ve = self.embedding_v(verbs)
     b, e = ve.size()   
     viz = self.noun_project(rep_n).view(b, 1, -1)
     viz = rep_n
 
     mr = self.encoding.max_roles()
     roles = Variable(self.get_roles(verbs).cuda()).view(b, mr)

     noun_pots = []
     noun_maxes = []
     mi = Variable(torch.LongTensor([self.encoding.n_nouns()]).cuda()).expand(b) #start symbol
     statep = None
     for x in range(0, mr):
       outputp = torch.cat([ve.view(b,1,-1), self.embedding_r(roles[:,x]).view(b,1,-1),self.embedding_n(mi).view(b, 1, -1),viz.view(b,1,-1)], dim=2)
       output, statex = self.noun_model(outputp, statep) 
       noun_pot = self.dropout(self.noun_linear(output.view(b,-1)))
       noun_pots.append(noun_pot)
       mv, mi = torch.max(noun_pot, dim = 1)
       noun_maxes.append(mi.view(b, 1))
       statep = statex

     max_i = torch.cat(noun_maxes, dim=1).view(b, mr)
     return max_i

   def forward_test_topk_verb(self, image, k):
     #seperate reps
     #rep_v = self.cnn_v(image)
     rep_n = self.cnn_n(image)
     rep_v = rep_n

     b, rl = rep_n.size()
     mr = self.encoding.max_roles()    
    
     v_pot = self.verb_linear(rep_v)
     verb_values, verb_indicies = torch.topk(v_pot, k, 1)
     return (verb_values.view(b, k), verb_indicies.view(b,k)) 

   def forward_test_topk(self, image, k):
     #seperate reps
     #rep_v = self.cnn_v(image)
     rep_n = self.cnn_n(image)
     rep_v = rep_n

     b, rl = rep_n.size()
     mr = self.encoding.max_roles()    
    
     v_pot = self.verb_linear(rep_v)
     verb_values, verb_indicies = torch.topk(v_pot, k, 1)
     #print(verb_indicies)
      
     n_verbs = k#self.encoding.n_verbs()     
     ve = self.embedding_v(verb_indicies.view(-1))
     _ , e = ve.size()

     roles = Variable(self.get_roles(verb_indicies.view(-1)).cuda()).view(b, n_verbs, mr)
     
     verbs = ve.view(b, n_verbs, e)#.view(1,n_verbs, e).expand(b, n_verbs, e).contiguous()
     #need to select from the role list
     #roles = self.role_list.view(1, n_verbs, -1).expand(b, n_verbs, mr).contiguous()
     viz = rep_n.view(b, 1 , -1).expand(b, n_verbs, rl).contiguous()

     vb = b*n_verbs

     noun_pots = []
     noun_maxes = []
     mi = Variable(torch.LongTensor([self.encoding.n_nouns()]).cuda()).expand(vb) #start symbol
 
     statep = None
     for x in range(0, mr):
       #print(verbs.view(vb,1,-1).size())
       #print(self.embedding_r(roles[:,:,x]).view(vb,1,-1).size())
       #print(self.embedding_n(mi).view(vb, 1, -1).size())
       #print(viz.view(vb,1,-1).size())

       outputp = torch.cat([verbs.view(vb,1,-1), self.embedding_r(roles[:,:,x]).view(vb,1,-1),self.embedding_n(mi).view(vb, 1, -1),viz.view(vb,1,-1)], dim=2)
       output, statex = self.noun_model(outputp, statep) 
       noun_pot = self.dropout(self.noun_linear(output.view(vb,-1)))
       noun_pots.append(noun_pot)
       mv, mi = torch.max(noun_pot, dim = 1)
       noun_maxes.append(mi.view(vb, 1))
       statep = statex

     noun_max_i = torch.cat(noun_maxes, dim=1).view(b,n_verbs, mr)
     #view to clarify the shape
     return (verb_values.view(b, k), verb_indicies.view(b,k), noun_max_i.view(b, k, mr)) 

   def loss(self, v_pot, n_pots, target):
     (b, nv, mr, n) = n_pots.size() 
     verbs = target[:,0].contiguous().view(b).contiguous()
     v = self.loss_v(v_pot, verbs)
     #print(n_pots.size())
     #print(n_pots)
     n_pots = n_pots.expand(b, 3, mr, n).contiguous().view(b*3*mr, n)
     #print(n_pots[0:6, :])
     #print(n_pots[6:12, :])
     n = self.loss_n(n_pots, target[:,2::2].contiguous().view(b*3*mr))  
     #print (v+n) 
     return v + 6*n #do we want to scale the noun loss?

def format_dict(d, s, p):
  rv = ""
  for (k,v) in d.items():
    if len(rv) > 0: rv += " , "
    rv+=p+str(k) + ":" + s.format(v*100)
  return rv

def predict_human_readable (dataset_loader, simple_dataset,  model, outdir, top_k):
  model.eval()  
  print ("predicting..." )
  mx = len(dataset_loader) 
  for i, (input, index) in enumerate(dataset_loader):
      print ("{}/{} batches".format(i+1,mx))
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      #(s_sorted, idx) = torch.sort(scores, 1, True)
      human = encoder.to_situation(predictions)
      (b,p,d) = predictions.size()
      for _b in range(0,b):
        items = []
        offset = _b *p
        for _p in range(0, p):
          items.append(human[offset + _p])
          items[-1]["score"] = scores.data[_b][_p]
        items = sorted(items, key = lambda x: -x["score"])[:top_k]
        name = simple_dataset.images[index[_b][0]].split(".")[:-1]
        name.append("predictions")
        outfile = outdir + ".".join(name)
        json.dump(items,open(outfile,"w"))


def compute_features(dataset_loader, simple_dataset,  model, outdir):
  model.eval()  
  print ("computing features..." )
  mx = len(dataset_loader) 
  for i, (input, index) in enumerate(dataset_loader):
      print ("{}/{} batches\r".format(i+1,mx)) ,
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      features  = model.forward_features(input_var).cpu().data
      b = index.size()[0]
      for _b in range(0,b):
        name = simple_dataset.images[index[_b][0]].split(".")[:-1]
        name.append("features")
        outfile = outdir + ".".join(name)
        torch.save(features[_b], outfile)
  print ("\ndone.")

def eval_model(dataset_loader, encoding, model, top1, top5, f = None):
    model.eval()
    print ("evaluating model...")
    #top1 = imSituTensorEvaluation(1, 3, encoding)
    #top5 = imSituTensorEvaluation(5, 3, encoding)
 
    mx = len(dataset_loader) 
    for i, (index, input, target) in enumerate(dataset_loader):
      print ("{}/{} batches\r".format(i+1,mx)) ,
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      target_var = torch.autograd.Variable(target.cuda(), volatile = True)
      if f is not None: target_var = f(encoding, target)

      #frame_predictions_given_verb  = model.forward(input_var,"given_verb", target[:,0].contiguous())
      _, frame_predictions_given_verb,_,_  = model.forward(input_var,"given_verb", target)
      (scores, verb_predictions)  = model.forward(input_var, "topk_verbs", 5)
     
      #(s_sorted, idx) = torch.sort(scores, 1, True)
      top1.add_point(target_var.data, verb_predictions.data, frame_predictions_given_verb)
      top5.add_point(target_var.data, verb_predictions.data, frame_predictions_given_verb)     
      
    print ("\ndone.")
    return (top1, top5) 

class verb_eval:
  def __init__(self, topk):
    self.score_cards = {}
    self.topk = topk
  
  def clear(self):
    self.score_cards = {}

  def add_point(self, encoded_reference, encoded_predictions, sorted_idx): 
    b = encoded_predictions.size()[0]
    for i in range(0,b):
      gt_v = encoded_reference[i][0] 
      verb_found = (torch.sum(sorted_idx[i][0:self.topk] == gt_v) == 1)
      sc_key = gt_v
      if sc_key not in self.score_cards:
        new_card = {"verb":0.0, "n_images": 0.0}
        self.score_cards[sc_key] = new_card
      _score_card = self.score_cards[sc_key]
      _score_card["n_images"] += 1
      if verb_found: _score_card["verb"] += 1
  
  def combine(self, rv, card):
    for (k,v) in card.items(): rv[k] += v

  def get_average_results(self):
    rv = {"verb":0.0}
    for (v, card) in self.score_cards.items():
      rv["verb"] += card["verb"] / card["n_images"]
    rv["verb"] /= len(self.score_cards)
    return rv

class noun_eval:
  def __init__(self, topk):
    self.score_cards = {}
    self.topk = topk
    self.p = 0
    self.pn = 0 
    self.r = 0
    self.rn = 0 

  def clear(self):
    self.p = 0
    self.pn = 0 
    self.r = 0
    self.rn = 0 

  #the ref will be b x n
  #the encoded predictions will be b x n x 2
  #the index will be b x n x 2 \in {1,0}
  def add_point(self, encoded_reference, encoded_predictions, sorted_idx): 
    (b,n,_) = encoded_predictions.size()
    for i in range(0,b):
      #print "--"
      #print sorted_idx[i,:,0]
      #print encoded_reference[i]
      #print torch.eq(sorted_idx[i,:,0], encoded_reference[i].data.cuda())
      pred = sorted_idx[i,:,0].cuda()
      ref = encoded_reference[i].cuda()
      self.p += torch.sum(torch.mul(pred==1,torch.eq(pred, ref) ))
      self.pn += torch.sum(pred == 1)
      self.r += torch.sum(torch.mul(ref==1,torch.eq(pred,ref) ))
      self.rn += torch.sum(ref==1) 
     # for j in range(0,n):
     #   gt = encoded_reference[i][j]
     #   found = (sorted_idx[i][j][0] == gt.data[0])
     # sc_key = j
     # if sc_key not in self.score_cards:
     #   new_card = {"noun":0.0, "n_images": 0.0}
     #   self.score_cards[sc_key] = new_card
     # _score_card = self.score_cards[sc_key]
     # _score_card["dn_images"] += 1
      #if found: 
      #  _score_card["noun"] += 1
      #  self.g += 1.0
  
  def combine(self, rv, card):
    for (k,v) in card.items(): rv[k] += v

  def get_average_results(self):
    rv = {}
    #for (v, card) in self.score_cards.items():
    #  rv["noun_micro"] += card["noun"] / card["n_images"]
    #rv["noun_micro"] /= len(self.score_cards)
    if self.pn == 0: rv["p"] = 0
    else: rv["p"] = (self.p + 0.0) / self.pn
    if self.rn == 0: rv["r"]  = 0 
    else: rv["r"] = (self.r + 0.0) / self.rn
    if rv["r"] == 0 and rv["p"] == 0: rv["f"] = 0
    else: rv["f"] = (rv["p"]*rv["r"])/( rv["p"] + rv["r"])
    return rv

class imSituSubsetTensorEvaluation():
  def __init__(self, topk, nref, encoding, bug = False):#image_group = {}):
    self.score_cards = {}
    self.topk = topk
    self.nref = nref
    self.encoding = encoding
    self.bug = bug
    #self.image_group = image_group
    
  def clear(self): 
    self.score_cards = {}

  def add_point(self, encoded_reference, encoded_predictions, sorted_idx, image_names = None, unk_symbol = 1, pad_symbol = 0):
    #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
    #encoded reference should be batch x 1+ references*roles,values (sorted) 
    (b,l) = encoded_predictions.size()
    for i in range(0,b):
      _pred = encoded_predictions[i]
      _ref = encoded_reference[i]
      _sorted_idx = sorted_idx[i]
      if image_names is not None: _image = image_names[i]

      lr = _ref.size()[0]     
      max_r = (lr - 1)/2/self.nref
      gt_v = _ref[0].item()

      if image_names is not None and _image in self.image_group: sc_key = (gt_v, self.image_group[_image])
      else: sc_key = (gt_v, "")
 
      if sc_key not in self.score_cards: 
        new_card = {"verb":0.0, "n_image":0.0, "value*":0.0, "n_value":0.0, "value-all*":0.0}
        self.score_cards[sc_key] = new_card
      
      v_roles = []
      for k in range(0, self.encoding.verb_nroles(gt_v)):
        _id = _ref[2*k + 1]
        #if _id == pad_symbol: break
        v_roles.append(_id)
      if len(v_roles) == 0: print( _ref  )  
      _score_card = self.score_cards[sc_key]
      _score_card["n_image"] += 1
      
      k = 0
      p_frame = None
      verb_found = (torch.sum(_sorted_idx[0:self.topk] == gt_v) == 1)
      if verb_found: _score_card["verb"] += 1
      p_frame = _pred#[0]#the gt verb will be in the first position 
      all_found = True
      exist_unk = False
      n = len(v_roles)
      #if n > 3: n = 3
      for k in range(0, n):
        _score_card["n_value"] += 1
        nv = p_frame[k]
        #print nv
        found = False
        for r in range(0,self.nref):
#          print nv
#          print _ref[1 + 2*max_r*r + 2*k+1]
          val = _ref[int(1 + 2*max_r*r + 2*k + 1)]
          #print self.encoding.id_r[v_roles[k]]
          
          #if val == pad_symbol and r == 0 and k < 2: exist_unk = True
          if self.bug and not exist_unk and val == pad_symbol and r == 0 and (self.encoding.id_r[v_roles[k]] == "place" or self.encoding.id_r[v_roles[k]] == "agent"):
            exist_unk = True
            break

#and (self.encoding.id_r[v_roles[k]] == "place" or self.encoding.id_r[v_roles[k]] == "agent") : exist_unk = True
          if (nv != pad_symbol and nv != unk_symbol and nv == val) :
            found = True
            break
        #potential bug
        if self.bug and exist_unk: #or nv == unk_symbol or nv == pad_symbol: 
          found = True
          _score_card["n_value"] -= 1
          _score_card["value*"] -= 1
        #if not found and not nv == unk_symbol: all_found = False
        #if not found and not exist_unk: all_found = False
        if not found: all_found = False
        if found: _score_card["value*"] += 1
#        print found
#      print all_found
      if all_found: _score_card["value-all*"] += 1
  
  def combine(self, rv, card):
    for (k,v) in card.items(): rv[k] += v

  def get_average_results(self, groups = []):
    #average across score cards.  
    rv = {"verb":0, "value*":0 , "value-all*":0}
    agg_cards = {}
    for (key, card) in self.score_cards.items():
      (v,g) = key
      if len(groups) == 0 or g in groups:
        if v not in agg_cards: 
          new_card = {"verb":0.0, "n_image":0.0, "value*":0.0, "n_value":0.0, "value-all*":0.0}
          agg_cards[v] = new_card
        self.combine(agg_cards[v], card)
    nverbs = len(agg_cards)
    for (v, card) in agg_cards.items():
      img = card["n_image"] 
      nvalue = card["n_value"]
      rv["verb"] += card["verb"]/img
      rv["value-all*"] += card["value-all*"]/img
      if nvalue > 0 : rv["value*"] += card["value*"]/(nvalue)
      
    rv["verb"] /= nverbs
    rv["value-all*"] /= nverbs 
    rv["value*"] /= nverbs

    return rv


def transform_noun(encoding,target):
          target2 = torch.LongTensor(target.size()[0], encoding.n_nouns()).zero_()
          #convert away from the one hot encoding
          bs = target.size()[0]   
          rs = encoding.max_roles()
          for b in range(0,bs):
            _t = set()
            for i in range(0,3):
              for r in range(0, rs):
                v = target[b][(2*rs)*i + 2*r + 2]
                if v != encoding.pad_symbol() : _t.add(v)
            #print _t
            for n in _t:
              target2[b][n] = 1
          return  Variable(target2)

def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model, encoding, optimizer, save_dir, timing = False, mode = 'joint', bug = False): 
    model.train()

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor = .1, verbose = True, min_lr=1e-6)

    time_all = time.time()

    pmodel = torch.nn.DataParallel(model, device_ids=device_array)
 
    if mode == "joint":  
      top1 = imSituSubsetTensorEvaluation(1, 3, encoding, bug)
      top5 = imSituSubsetTensorEvaluation(5, 3, encoding, bug)
      top1_e = imSituTensorV2Evaluation(1, 3, encoding)
      top5_e = imSituTensorV2Evaluation(5, 3, encoding)
    elif mode == "noun":
      top1 = noun_eval(1)
      top5 = noun_eval(1)#not sure what top 5 means for binary ;)
      top1_e = noun_eval(1)
      top5_e = noun_eval(1)
    elif mode == "verb":
      top1 = verb_eval(1)
      top5 = verb_eval(5)
      top1_e = verb_eval(1)
      top5_e = verb_eval(5)

    loss_total = 0 
    print_freq = 10
    total_steps = 0
    avg_scores = []
    
    model.set_train_mode(mode)
    lr_steps = 0
    for k in range(0,max_epoch):  
      for i, (index, input, target) in enumerate(train_loader):
        total_steps += 1
   
        t0 = time.time()
        t1 = time.time() 
      
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())
      
        (scores, predictions, noun_pots, verb_pots) = pmodel(input_var, "train", target_var)
        loss = model.loss(verb_pots, noun_pots, target_var)
        (s_sorted, idx) = torch.sort(scores, len(scores.size())-1, True)
     
        #print norm 
        if timing: print("loss time = {}".format(time.time() - t1))
        optimizer.zero_grad()
        t1 = time.time()
        loss.backward()
        #print loss
        if timing: print("backward time = {}".format(time.time() - t1))
        optimizer.step()
        loss_total += loss.data.item()
        #score situation
        t2 = time.time() 
        top1.add_point(target_var.data, predictions.data, idx.data)
        top5.add_point(target_var.data, predictions.data, idx.data)
     
        if timing: print("eval time = {}".format(time.time() - t2))
        if timing: print("batch time = {}".format(time.time() - t0))
        if total_steps % print_freq == 0:
           
           top1_a = top1.get_average_results()
           top5_a = top5.get_average_results()
           #top1.clear() 
           #top5.clear()
           print( "{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(total_steps-1,k,i, format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a,"{:.2f}","5-"), loss.data[0], loss_total / ((total_steps-1)%eval_frequency) , (time.time() - time_all)/ ((total_steps-1)%eval_frequency)))
        if total_steps % eval_frequency == 0:
          print ("eval...")    
          etime = time.time()
          if mode == 'noun': eval_model(dev_loader, encoding,pmodel,top1_e, top5_e, transform_noun )
          elif mode == 'verb' or mode == 'joint': eval_model(dev_loader, encoding, pmodel, top1_e, top5_e, None)
          model.train() 
          print ("... done after {:.2f} s".format(time.time() - etime))
          top1_a = top1_e.get_average_results()
          top5_a = top5_e.get_average_results()

          if mode == "joint" : 
            avg_score = top1_a["verb"] + top5_a["verb"] + top5_a["value*"] + top5_a["value-all*"] #top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a["value-all"] 
            avg_score /= 4
          elif mode == "verb":
            avg_score = (top1_a["verb"] + top5_a["verb"])/2 
          elif mode == "noun":
            avg_score = top1_a["f"]
          print ("Dev {} average :{:.2f} {} {}".format(total_steps-1, avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))
            
          #scheduler.step(avg_score)
          if len(avg_scores) >= 1 : pmax = max(avg_scores)
          avg_scores.append(avg_score)
          maxv = max(avg_scores)
          torch.save(model.state_dict(), save_dir + "/{0}.model".format(maxv))   
          if len(avg_scores) > 1: print("diff = {}".format(maxv - pmax))
          if maxv == avg_scores[-1] and (len(avg_scores) < 2 or (maxv - pmax > .001)) : 
          #  torch.save(model.state_dict(), save_dir + "/{0}.model".format(maxv))   
            print("continuing, new best model saved! {0}".format(maxv))
          else:
            lr_steps += 1
            print("loading old model (we never accept)")
            model.load_state_dict(torch.load(save_dir+"/{}.model".format(maxv)))
            if lr_steps == 1:
               print("resetting fine tuning TRUE and reducing lr")
               model.set_ft(True) 
               model.attention = True
               optimizer = optim.Adam(model.opt_parameters(), lr = args.learning_rate*.1 , weight_decay = args.weight_decay)
            else:
              for param_group in optimizer.param_groups:
                param_group['lr'] = .1*param_group['lr']
                print(param_group['lr'])
           
          top1.clear() 
          top5.clear()
          top1_e.clear()
          top5_e.clear() 
          loss_total = 0
          time_all = time.time()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="imsitu Situation CRF. Training, evaluation, prediction and features.") 
  parser.add_argument("--command", choices = ["train", "eval", "predict", "features"], required = True)
  parser.add_argument("--output_dir", help="location to put output, such as models, features, predictions")
  parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
  parser.add_argument("--dataset_dir", default="./", help="location of train.json, dev.json, ect.") 
  parser.add_argument("--weights_file", help="the model to start from")
  parser.add_argument("--encoding_file", help="a file corresponding to the encoder")
  parser.add_argument("--cnn_type", choices=["vgg", "resnet_34", "resnet_50", "resnet_101"], default="resnet_34", help="the cnn to initilize the crf with") 
  parser.add_argument("--global_dropout", default=.5, help="dropout for rep layer", type=float)
  parser.add_argument("--hidden_rep", default=1024, help="hidden dim size for lstm", type=int)
  parser.add_argument("--noun_rep", default=512, help="noun rep size", type=int)
  parser.add_argument("--role_rep", default=64, help="role rep size", type=int)
  parser.add_argument("--verb_rep", default=256, help="verb rep size", type=int)
  parser.add_argument("--norm_vision", default=0, help="norm the visual vector", type=int)
  parser.add_argument("--feed_self", default=0, help="use prediction during training", type=int)
  parser.add_argument("--fine_tune_cnn", default=1, help="fine tune the cnn", type = int)
  parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
  parser.add_argument("--learning_rate", default=1e-5, help="learning rate for ADAM", type=float)
  parser.add_argument("--weight_decay", default=5e-4, help="learning rate decay for ADAM", type=float)  
  parser.add_argument("--eval_frequency", default=500, help="evaluate on dev set every N training steps", type=int) 
  parser.add_argument("--training_epochs", default=20, help="total number of training epochs", type=int)
  parser.add_argument("--eval_file", default="dev.json", help="the dataset file to evaluate on, ex. dev.json test.json")
  parser.add_argument("--top_k", default="10", type=int, help="topk to use for writing predictions to file")
  parser.add_argument("--mode", default="verb" , help="training mode")
  parser.add_argument("--ngpus", default=1, type=int, help="number of gpus")
  parser.add_argument("--unk_threshold", default = 25, type = int, help = "unk threshold")
  parser.add_argument("--depth", default  = 1, type = int , help = "gnn depth")
  parser.add_argument("--pretrained_noun_file", default = "", help="file for pretrained noun file")
  parser.add_argument("--pretrained_verb_file", default = "", help="file for pretrained verb file")
  parser.add_argument("--bug_eval", default= 0, type = int, help="whether to bug the eval or not")
  parser.add_argument("--attention", default= 0, type = int, help="whether to use spatial attention or not")
  args = parser.parse_args()
  global_dropout = args.global_dropout
  print (args)
  if args.command == "train":
    print ("command = training")
    train_set = json.load(open(args.dataset_dir+"/train.json"))
    dev_set = json.load(open(args.dataset_dir+"/dev.json"))

    if args.encoding_file is None:
      print ("creating new encoder") 
      encoder = imSituVerbRoleNounEncoder(train_set, unk = args.unk_threshold)
      torch.save(encoder, args.output_dir + "/encoder")
    else:
      encoder = torch.load(args.encoding_file)
  
    model = rnn_predictor_spatial(encoder, cnn_type = args.cnn_type, ngpus=args.ngpus, hidden_size = args.hidden_rep, v_rep_size = args.verb_rep, r_rep_size = args.role_rep, n_rep_size = args.noun_rep, norm_vision = args.norm_vision, feed_self = args.feed_self, fine_tune_cnn = args.fine_tune_cnn, attention = args.attention)
    
    if args.weights_file is not None:
      model.load_state_dict(torch.load(args.weights_file))
    
    dataset_train = imSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
    dataset_dev = imSituSituation(args.image_dir, dev_set, encoder, model.dev_preprocess())

    ngpus = args.ngpus
    device_array = [i for i in range(0,ngpus)]
    batch_size = args.batch_size*ngpus

    train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 3) 
    dev_loader  = torch.utils.data.DataLoader(dataset_dev, batch_size = int(batch_size), shuffle = True, num_workers = 3) 

    model.cuda()
    optimizer = optim.Adam(model.opt_parameters(), lr = args.learning_rate , weight_decay = args.weight_decay)
    train_model(args.training_epochs, args.eval_frequency, train_loader, dev_loader, model, encoder, optimizer, args.output_dir, mode = args.mode, bug = args.bug_eval)
  
  elif args.command == "eval":
    print ("command = evaluating")
    eval_file = json.load(open(args.dataset_dir + "/" + args.eval_file))  
      
    if args.encoding_file is None: 
      print ("expecting encoder file to run evaluation")
      exit()
    else:
      encoder = torch.load(args.encoding_file)
    print ("creating model..." )
    model = baseline_crf(encoder, cnn_type = args.cnn_type)
    
    if args.weights_file is None:
      print ("expecting weight file to run features")
      exit()
    
    print ("loading model weights...")
    model.load_state_dict(torch.load(args.weights_file))
    model.cuda()
    
    dataset = imSituSituation(args.image_dir, eval_file, encoder, model.train_preprocess())
    loader  = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 3) 

    (top1, top5) = eval_model(loader, encoder, model)    
    top1_a = top1.get_average_results()
    top5_a = top5.get_average_results()

    avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
    avg_score /= 8

    print ("Average :{:.2f} {} {}".format(avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))
       
  elif args.command == "features":
    print ("command = features")
    if args.encoding_file is None: 
      print ("expecting encoder file to run features")
      exit()
    else:
      encoder = torch.load(args.encoding_file)
 
    print ("creating model...")
    model = baseline_crf(encoder, cnn_type = args.cnn_type)
    
    if args.weights_file is None:
      print ("expecting weight file to run features")
      exit()
    
    print ("loading model weights...")
    model.load_state_dict(torch.load(args.weights_file))
    model.cuda()
    
    folder_dataset = imSituSimpleImageFolder(args.image_dir, model.dev_preprocess())
    image_loader  = torch.utils.data.DataLoader(folder_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 3) 

    compute_features(image_loader, folder_dataset, model, args.output_dir)    

  elif args.command == "predict":
    print ("command = predict")
    if args.encoding_file is None: 
      print ("expecting encoder file to run features")
      exit()
    else:
      encoder = torch.load(args.encoding_file)
 
    print ("creating model..." )
    model = baseline_crf(encoder, cnn_type = args.cnn_type)
 
    if args.weights_file is None:
      print ("expecting weight file to run features")
      exit()
    
    print ("loading model weights...")
    model.load_state_dict(torch.load(args.weights_file))
    model.cuda()

    folder_dataset = imSituSimpleImageFolder(args.image_dir, model.dev_preprocess())
    image_loader  = torch.utils.data.DataLoader(folder_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 3) 
    
    predict_human_readable(image_loader, folder_dataset, model, args.output_dir, args.top_k)
