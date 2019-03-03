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
from imsitu import imSituVerbRoleLocalNounEncoder 
from imsitu import imSituTensorEvaluation 
from imsitu import imSituSituation 
from imsitu import imSituSimpleImageFolder
from utils import initLinear
import json

class vgg_modified(nn.Module):
  def __init__(self):
    super(vgg_modified,self).__init__()
    self.vgg = tv.models.vgg16(pretrained=True)
    self.vgg_features = self.vgg.features
    #self.classifier = nn.Sequential(
            #nn.Dropout(),
    self.lin1 = nn.Linear(512 * 7 * 7, 1024)
    self.relu1 = nn.ReLU(True)
    self.dropout1 = nn.Dropout()
    self.lin2 =  nn.Linear(1024, 1024)
    self.relu2 = nn.ReLU(True)
    self.dropout2 = nn.Dropout()

    initLinear(self.lin1)
    initLinear(self.lin2)
  
  def rep_size(self): return 1024

  def forward(self,x):
    return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))

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
 
 
class resnet_modified_small(nn.Module):
 def __init__(self):
    super(resnet_modified_small, self).__init__()
    self.resnet = tv.models.resnet34(pretrained=True)
    #probably want linear, relu, dropout
    self.linear = nn.Linear(7*7*512, 1024)
    self.dropout2d = nn.Dropout2d(.5)
    self.dropout = nn.Dropout(.5)
    self.relu = nn.LeakyReLU()
    initLinear(self.linear)

 def base_size(self): return 512
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

        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))
      
class baseline_crf(nn.Module):
   def train_preprocess(self): return self.train_transform
   def dev_preprocess(self): return self.dev_transform

   #these seem like decent splits of imsitu, freq = 0,50,100,282 , prediction type can be "max_max" or "max_marginal"
   def __init__(self, encoding, splits = [50,100,283], prediction_type = "max_max", ngpus = 1, cnn_type = "resnet_101"):
     super(baseline_crf, self).__init__() 
     
     self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     self.train_transform = tv.transforms.Compose([
            tv.transforms.Scale(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

     self.dev_transform = tv.transforms.Compose([
            tv.transforms.Scale(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

     self.broadcast = []
     self.nsplits = len(splits)
     self.splits = splits
     self.encoding = encoding
     self.prediction_type = prediction_type
     self.n_verbs = encoding.n_verbs()
     self.split_vr = {}
     self.v_roles = {}
     #cnn
     print(cnn_type)
     if cnn_type == "resnet_101" : self.cnn = resnet_modified_large()
     elif cnn_type == "resnet_50": self.cnn = resnet_modified_medium()
     elif cnn_type == "resnet_34": self.cnn = resnet_modified_small()
     else: 
       print("unknown base network")
       exit()
     self.rep_size = self.cnn.rep_size()
     for s in range(0,len(splits)): self.split_vr[s] = []

     #sort by length
     remapping = []
     for (vr, ns) in encoding.vr_id_n.items(): remapping.append((vr, len(ns)))
     
     #find the right split
     for (vr, l) in remapping:
       i = 0
       for s in splits:
         if l <= s: break
         i+=1  
       _id = (i, vr)
       self.split_vr[i].append(_id)
     total = 0 
     for (k,v) in self.split_vr.items():
       #print "{} {} {}".format(k, len(v), splits[k]*len(v))
       total += splits[k]*len(v) 
     #print "total compute : {}".format(total) 
     
     #keep the splits sorted by vr id, to keep the model const w.r.t the encoding 
     for i in range(0,len(splits)):
       s = sorted(self.split_vr[i], key = lambda x: x[1])
       self.split_vr[i] = []
       #enumerate?
       for (x, vr) in s: 
         _id = (x,len(self.split_vr[i]), vr)
         self.split_vr[i].append(_id)    
         (v,r) = encoding.id_vr[vr]
         if v not in self.v_roles: self.v_roles[v] = []
         self.v_roles[v].append(_id)
    
     #create the mapping for grouping the roles back to the verbs later       
     max_roles = encoding.max_roles()

     #need a list that is nverbs by 6
     self.v_vr = [ 0 for i in range(0, self.encoding.n_verbs()*max_roles) ]
     splits_offset = []
     for i in range(0,len(splits)):
       if i == 0: splits_offset.append(0)
       else: splits_offset.append(splits_offset[-1] + len(self.split_vr[i-1]))
    
     #and we need to compute the position of the corresponding roles, and pad with the 0 symbol
     for i in range(0, self.encoding.n_verbs()):
       offset = max_roles*i
       roles = sorted(self.v_roles[i] , key=lambda x: x[2]) #stored in role order
       self.v_roles[i] = roles
       k = 0
       for (s, pos, r) in roles:
         #add one to account of the 0th element being the padding
         self.v_vr[offset + k] = splits_offset[s] + pos + 1
         k+=1
       #pad
       while k < max_roles:
         self.v_vr[offset + k] = 0
         k+=1
     
     gv_vr = Variable(torch.LongTensor(self.v_vr).cuda())#.view(self.encoding.n_verbs(), -1) 
     for g in range(0,ngpus):
       self.broadcast.append(Variable(torch.LongTensor(self.v_vr).cuda(g)))
     self.v_vr = gv_vr
     #print self.v_vr

     #verb potential
     self.linear_v = nn.Linear(self.rep_size, self.encoding.n_verbs())
     #verb-role-noun potentials
     self.linear_vrn = nn.ModuleList([ nn.Linear(self.rep_size, splits[i]*len(self.split_vr[i])) for i in range(0,len(splits))])
     self.total_vrn = 0
     for i in range(0, len(splits)): self.total_vrn += splits[i]*len(self.split_vr[i])
     print("total encoding vrn : {0}, with padding in {1} groups : {2}".format(encoding.n_verbrolenoun(), self.total_vrn, len(splits))) 

     #initilize everything
     initLinear(self.linear_v)
     for _l in self.linear_vrn: initLinear(_l)
     self.mask_args()

   def mask_args(self):
     #go through the and set the weights to negative infinity for out of domain items     
     neg_inf = float("-infinity")
     for v in range(0, self.encoding.n_verbs()):
       for (s, pos, r) in self.v_roles[v]:
         linear = self.linear_vrn[s] 
         #get the offset
#         print self.splits
         start = self.splits[s]*pos+len(self.encoding.vr_n_id[r])
         end = self.splits[s]*(pos+1)
         for k in range(start,end):
           linear.bias.data[k] = -100 #neg_inf
            
   #expects a list of vectors, BxD
   #returns the max index of every vector, max value of each vector and the log_sum_exp of the vector
   def log_sum_exp(self,vec):
     max_score, max_i = torch.max(vec,1)
     max_score_broadcast = max_score.view(-1,1).expand(vec.size())
     return (max_i , max_score,  max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast),1)))

   def forward_max(self, images):
     (_,_,_,_,scores, values) = self.forward(images)
     return (scores, values)

   def forward_features(self, images):
     return self.cnn(images)
   
   def forward(self, image):
     batch_size = image.size()[0]

     rep = self.cnn(image)
     #print self.rep_size
     #print batch_size
     v_potential = self.linear_v(rep)
     
     vrn_potential = []
     vrn_marginal = []
     vr_max = []
     vr_maxi = []
     #first compute the norm
     #step 1 compute the verb-role marginals
     #this loop allows a memory/parrelism tradeoff. 
     #To use less memory but achieve less parrelism, increase the number of groups
     for i,vrn_group in enumerate(self.linear_vrn): 
       #linear for the group
       _vrn = vrn_group(rep).view(-1, self.splits[i])
       
       _vr_maxi, _vr_max ,_vrn_marginal = self.log_sum_exp(_vrn)
       _vr_maxi = _vr_maxi.view(-1, len(self.split_vr[i]))
       _vr_max = _vr_max.view(-1, len(self.split_vr[i]))
       _vrn_marginal = _vrn_marginal.view(-1, len(self.split_vr[i]))
     
       vr_maxi.append(_vr_maxi)
       vr_max.append(_vr_max)
       vrn_potential.append(_vrn.view(batch_size, -1, self.splits[i]))
       vrn_marginal.append(_vrn_marginal)
     
     #concat role groups with the padding symbol 
     zeros = Variable(torch.zeros(batch_size, 1).cuda()) #this is the padding 
     zerosi = Variable(torch.LongTensor(batch_size,1).zero_().cuda())
     vrn_marginal.insert(0, zeros)
     vr_max.insert(0,zeros)
     vr_maxi.insert(0,zerosi)

     #print vrn_marginal
     vrn_marginal = torch.cat(vrn_marginal, 1)
     vr_max = torch.cat(vr_max,1)
     vr_maxi = torch.cat(vr_maxi,1)     

     #print vrn_marginal
     #step 2 compute verb marginals
     #we need to reorganize the role potentials so it is BxVxR
     #gather the marginals in the right way
     v_vr = self.broadcast[torch.cuda.current_device()] 
     vrn_marginal_grouped = vrn_marginal.index_select(1,v_vr).view(batch_size,self.n_verbs,self.encoding.max_roles())
     vr_max_grouped = vr_max.index_select(1,v_vr).view(batch_size, self.n_verbs, self.encoding.max_roles()) 
     vr_maxi_grouped = vr_maxi.index_select(1,v_vr).view(batch_size, self.n_verbs, self.encoding.max_roles())
     
     # product ( sum since we are in log space )
     v_marginal = torch.sum(vrn_marginal_grouped, 2).view(batch_size, self.n_verbs) + v_potential
    
     #step 3 compute the final sum over verbs
     _, _ , norm  = self.log_sum_exp(v_marginal)
     #compute the maxes

     #max_max probs
     v_max = torch.sum(vr_max_grouped,2).view(batch_size, self.n_verbs) + v_potential #these are the scores
     #we don't actually care, we want a max prediction per verb
     #max_max_vi , max_max_v_score = max(v_max,1)
     #max_max_prob = exp(max_max_v_score - norm)
     #max_max_vrn_i = vr_maxi_grouped.gather(1,max_max_vi.view(batch_size,1,1).expand(batch_size,1,self.max_roles))

     #offset so we can use index select... is there a better way to do this?
     #max_marginal probs 
     #max_marg_vi , max_marginal_verb_score = max(v_marginal, 1)
     #max_marginal_prob = exp(max_marginal_verb_score - norm)
     #max_marg_vrn_i = vr_maxi_grouped.gather(1,max_marg_vi.view(batch_size,1,1).expand(batch_size,1,self.max_roles))
     
     #this potentially does not work with parrelism, in which case we should figure something out 
     if self.prediction_type == "max_max":
       rv = (rep, v_potential, vrn_potential, norm, v_max, vr_maxi_grouped) 
     elif self.prediction_type == "max_marginal":
       rv = (rep, v_potential, vrn_potential, norm, v_marginal, vr_maxi_grouped) 
     else:
       print("unkown inference type")
       rv = ()
     return rv

  
   #computes log( (1 - exp(x)) * (1 - exp(y)) ) =  1 - exp(y) - exp(x) + exp(y)*exp(x) = 1 - exp(V), so V=  log(exp(y) + exp(x) - exp(x)*exp(y))
   #returns the the log of V 
   def logsumexp_nx_ny_xy(self, x, y):
     #_,_, v = self.log_sum_exp(torch.cat([x, y, torch.log(torch.exp(x+y))]).view(1,3))
     if x > y: 
       return torch.log(torch.exp(y-x) + 1 - torch.exp(y) + 1e-8) + x
     else:
       return torch.log(torch.exp(x-y) + 1 - torch.exp(x) + 1e-8) + y

   def sum_loss(self, v_potential, vrn_potential, norm, situations, n_refs):
     #compute the mil losses... perhaps this should be a different method to facilitate parrelism?
     batch_size = v_potential.size()[0]
     mr = self.encoding.max_roles()
     for i in range(0,batch_size):
       _norm = norm[i]
       _v = v_potential[i]
       _vrn = []
       _ref = situations[i]
       for pot in vrn_potential: _vrn.append(pot[i])
       for r in range(0,n_refs):
         v = _ref[0]
         pots = _v[v]
         for (pos,(s, idx, rid)) in enumerate(self.v_roles[v]):
           pots = pots + _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
         if pots.data[0] > _norm.data[0]: 
           print("inference error")
           print(pots)
           print (_norm)
         if i == 0 and r == 0: loss = pots-_norm
         else: loss = loss + pots - _norm
     return -loss/(batch_size*n_refs)

   def mil_loss(self, v_potential, vrn_potential, norm,  situations, n_refs): 
     #compute the mil losses... perhaps this should be a different method to facilitate parrelism?
     batch_size = v_potential.size()[0]
     mr = self.encoding.max_roles()
     for i in range(0,batch_size):
       _norm = norm[i]
       _v = v_potential[i]
       _vrn = []
       _ref = situations[i]
       for pot in vrn_potential: _vrn.append(pot[i])
       for r in range(0,n_refs):
         v = _ref[0]
         pots = _v[v]
         for (pos,(s, idx, rid)) in enumerate(self.v_roles[v.item()]):
       #    print _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
#_vrn[s][idx][
           pots = pots + _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
         if pots.item() > _norm.item(): 
           print("inference error")
           print(pots)
           print(_norm)
         if r == 0: _tot = pots-_norm 
         else : _tot = self.logsumexp_nx_ny_xy(_tot, pots-_norm)
       if i == 0: loss = _tot
       else: loss = loss + _tot
     return -loss/batch_size

def format_dict(d, s, p):
    rv = ""
    for (k,v) in d.items():
      if len(rv) > 0: rv += " , "
      rv+=p+str(k) + ":" + s.format(v*100)
    return rv

def predict_human_readable (dataset_loader, simple_dataset,  model, outdir, top_k):
  model.eval()  
  print("predicting...")
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
          items[-1]["score"] = scores.data[_b][_p].item()
        items = sorted(items, key = lambda x: -x["score"])[:top_k]
        name = simple_dataset.images[index[_b][0]].split(".")[:-1]
        name.append("predictions")
        outfile = outdir + ".".join(name)
        json.dump(items,open(outfile,"w"))


def compute_features(dataset_loader, simple_dataset,  model, outdir):
  model.eval()  
  print ("computing features...")
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

def eval_model(dataset_loader, encoding, model):
    model.eval()
    print ("evaluating model...")
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
 
    mx = len(dataset_loader) 
    for i, (index, input, target) in enumerate(dataset_loader):
      print ("{}/{} batches\r".format(i+1,mx)) ,
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      target_var = torch.autograd.Variable(target.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      (s_sorted, idx) = torch.sort(scores, 1, True)
      top1.add_point(target, predictions.data, idx.data)
      top5.add_point(target, predictions.data, idx.data)
      
    print ("\ndone.")
    return (top1, top5) 

def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model, encoding, optimizer, save_dir, timing = False): 
    model.train()

    time_all = time.time()

    pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
    loss_total = 0 
    print_freq = 10
    total_steps = 0
    avg_scores = []
  
    for k in range(0,max_epoch):  
      for i, (index, input, target) in enumerate(train_loader):
        total_steps += 1
   
        t0 = time.time()
        t1 = time.time() 
      
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())
        (_,v,vrn,norm,scores,predictions)  = pmodel(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)
        #print norm 
        if timing : print ("forward time = {}".format(time.time() - t1))
        optimizer.zero_grad()
        t1 = time.time()
        loss = model.mil_loss(v,vrn,norm, target, 3)
        if timing: print ("loss time = {}".format(time.time() - t1))
        t1 = time.time()
        loss.backward()
        #print loss
        if timing: print ("backward time = {}".format(time.time() - t1))
        optimizer.step()
        loss_total += loss.item()
        #score situation
        t2 = time.time() 
        top1.add_point(target, predictions.data, idx.data)
        top5.add_point(target, predictions.data, idx.data)
     
        if timing: print ("eval time = {}".format(time.time() - t2))
        if timing: print ("batch time = {}".format(time.time() - t0))
        if total_steps % print_freq == 0:
           top1_a = top1.get_average_results()
           top5_a = top5.get_average_results()
           print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(total_steps-1,k,i, format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a,"{:.2f}","5-"), loss.item(), loss_total / ((total_steps-1)%eval_frequency) , (time.time() - time_all)/ ((total_steps-1)%eval_frequency)))
        if total_steps % eval_frequency == 0:
          print ("eval..."    )
          etime = time.time()
          (top1, top5) = eval_model(dev_loader, encoding, model)
          model.train() 
          print ("... done after {:.2f} s".format(time.time() - etime))
          top1_a = top1.get_average_results()
          top5_a = top5.get_average_results()

          avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
          avg_score /= 8

          print ("Dev {} average :{:.2f} {} {}".format(total_steps-1, avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))
          
          avg_scores.append(avg_score)
          maxv = max(avg_scores)

          if maxv == avg_scores[-1]: 
            torch.save(model.state_dict(), save_dir + "/{0}.model".format(maxv))   
            print ("new best model saved! {0}".format(maxv))

          top1 = imSituTensorEvaluation(1, 3, encoding)
          top5 = imSituTensorEvaluation(5, 3, encoding)
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
  parser.add_argument("--cnn_type", choices=["resnet_34", "resnet_50", "resnet_101"], default="resnet_101", help="the cnn to initilize the crf with") 
  parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
  parser.add_argument("--learning_rate", default=1e-5, help="learning rate for ADAM", type=float)
  parser.add_argument("--weight_decay", default=5e-4, help="learning rate decay for ADAM", type=float)  
  parser.add_argument("--eval_frequency", default=500, help="evaluate on dev set every N training steps", type=int) 
  parser.add_argument("--training_epochs", default=20, help="total number of training epochs", type=int)
  parser.add_argument("--eval_file", default="dev.json", help="the dataset file to evaluate on, ex. dev.json test.json")
  parser.add_argument("--top_k", default="10", type=int, help="topk to use for writing predictions to file")

  args = parser.parse_args()
  if args.command == "train":
    print ("command = training")
    train_set = json.load(open(args.dataset_dir+"/train.json"))
    dev_set = json.load(open(args.dataset_dir+"/dev.json"))

    if args.encoding_file is None: 
      encoder = imSituVerbRoleLocalNounEncoder(train_set)
      torch.save(encoder, args.output_dir + "/encoder")
    else:
      encoder = torch.load(args.encoding_file)
  
    ngpus = 1
    model = baseline_crf(encoder, cnn_type = args.cnn_type, ngpus = ngpus)
    
    if args.weights_file is not None:
      model.load_state_dict(torch.load(args.weights_file))
    
    dataset_train = imSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
    dataset_dev = imSituSituation(args.image_dir, dev_set, encoder, model.dev_preprocess())

    device_array = [i for i in range(0,ngpus)]
    batch_size = args.batch_size*ngpus

    train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 3) 
    dev_loader  = torch.utils.data.DataLoader(dataset_dev, batch_size = batch_size, shuffle = True, num_workers = 3) 

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate , weight_decay = args.weight_decay)
    train_model(args.training_epochs, args.eval_frequency, train_loader, dev_loader, model, encoder, optimizer, args.output_dir)
  
  elif args.command == "eval":
    print ("command = evaluating")
    eval_file = json.load(open(args.dataset_dir + "/" + args.eval_file))  
      
    if args.encoding_file is None: 
      print ("expecting encoder file to run evaluation")
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
    
    dataset = imSituSituation(args.image_dir, eval_file, encoder, model.dev_preprocess())
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
