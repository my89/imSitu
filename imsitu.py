import torch as torch
import torch.utils.data as data
import json
from PIL import Image, ImageOps
import PIL
import os
import os.path

class imSituTensorV2Evaluation():
  def __init__(self, topk, nref, encoding = None, image_group = {}):
    self.score_cards = {}
    self.topk = topk
    self.nref = nref
    self.image_group = image_group
    self.encoding = encoding
    
  def clear(self): 
    self.score_cards = {}
    
  def add_point(self, encoded_reference, encoded_prediction_verbs, encoded_predictions_given_gt_verb, image_names = None, unk_symbol = 1, pad_symbol = 0):
    #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
    #encoded reference should be batch x 1+ references*roles,values (sorted) 
    #print(encoded_predictions.size())
    (b,_) = encoded_reference.size()
    for i in range(0,b):
      #_pred = encoded_predictions_nouns[i]
      _ref = encoded_reference[i]
      _sorted_idx = encoded_prediction_verbs[i]

      if image_names is not None: _image = image_names[i]

      lr = _ref.size()[0]     
      max_r = int((lr - 1)/2/self.nref)

      gt_v = _ref[0]
      if image_names is not None and _image in self.image_group: sc_key = (gt_v, self.image_group[_image])
      else: sc_key = (gt_v, "")
 
      if sc_key not in self.score_cards: 
        new_card = {"verb":0.0, "n_image":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
        self.score_cards[sc_key] = new_card
      
      v_roles = self.encoding.verb_roles(gt_v.item())
 
      _score_card = self.score_cards[sc_key]
      _score_card["n_image"] += 1
      _score_card["n_value"] += len(v_roles)
     
      k = 0
      p_frame = None
      verb_found = (torch.sum(_sorted_idx[0:self.topk] == gt_v) == 1)
      if verb_found: _score_card["verb"] += 1
      p_frame = encoded_predictions_given_gt_verb[i]  
      all_found = True

      for k in range(0, len(v_roles)):
        nv = p_frame[k]
        found = False
        for r in range(0,self.nref):
          val = _ref[1 + 2*max_r*r + 2*k + 1]
          if nv != unk_symbol and nv == val :
            found = True
            break
        if not found: all_found = False
        if found and verb_found: _score_card["value"] += 1
        if found: _score_card["value*"] += 1
      if all_found and verb_found: _score_card["value-all"] += 1
      if all_found: _score_card["value-all*"] += 1
  
  def combine(self, rv, card):
    for (k,v) in card.items(): rv[k] += v

  def get_average_results(self, groups = []):
    #average across score cards.  
    rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
    agg_cards = {}
    for (key, card) in self.score_cards.items():
      (v,g) = key
      if len(groups) == 0 or g in groups:
        if v not in agg_cards: 
          new_card = {"verb":0.0, "n_image":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
          agg_cards[v] = new_card
        self.combine(agg_cards[v], card)
    nverbs = len(agg_cards)
    for (v, card) in agg_cards.items():
      img = card["n_image"] 
      nvalue = card["n_value"]
      rv["verb"] += card["verb"]/img
      rv["value-all"] += card["value-all"]/img
      rv["value-all*"] += card["value-all*"]/img
      rv["value"] += card["value"]/nvalue
      rv["value*"] += card["value*"]/nvalue
      
    rv["verb"] /= nverbs
    rv["value-all"] /= nverbs
    rv["value-all*"] /= nverbs 
    rv["value"] /= nverbs
    rv["value*"] /= nverbs

    return rv





class imSituTensorEvaluation():
  def __init__(self, topk, nref, encoding = None, image_group = {}):
    self.score_cards = {}
    self.topk = topk
    self.nref = nref
    self.image_group = image_group
    self.encoding = encoding
    
  def clear(self): 
    self.score_cards = {}
    
  def add_point(self, encoded_reference, encoded_predictions, sorted_idx, image_names = None, unk_symbol = 1, pad_symbol = 0):
    #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
    #encoded reference should be batch x 1+ references*roles,values (sorted) 
    #print(encoded_predictions.size())
    (b,tv,l) = encoded_predictions.size()
    for i in range(0,b):
      _pred = encoded_predictions[i]
      _ref = encoded_reference[i]
      _sorted_idx = sorted_idx[i]
      if image_names is not None: _image = image_names[i]

      lr = _ref.size()[0]     
      max_r = int((lr - 1)/2/self.nref)

      gt_v = _ref[0]
      if image_names is not None and _image in self.image_group: sc_key = (gt_v, self.image_group[_image])
      else: sc_key = (gt_v, "")
 
      if sc_key not in self.score_cards: 
        new_card = {"verb":0.0, "n_image":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
        self.score_cards[sc_key] = new_card
      
      #v_roles = []
      #for k in range(0,max_r):
      #  _id = _ref[2*k + 2]
      #  if _id == pad_symbol: break
      #  v_roles.append(_id)
      #if len(v_roles) == 0:
      #   print (_ref  ) 
      #   print("error")  
      v_roles = self.encoding.verb_roles(gt_v.item())
 
      _score_card = self.score_cards[sc_key]
      _score_card["n_image"] += 1
      _score_card["n_value"] += len(v_roles)
     
      k = 0
      p_frame = None
      verb_found = (torch.sum(_sorted_idx[0:self.topk] == gt_v) == 1)
      if verb_found: _score_card["verb"] += 1
      p_frame = _pred[gt_v]  
      all_found = True
    #  if i == 0 and self.topk == 1: 
    #    print p_frame
 #       print _ref
#      print len(v_roles)
      for k in range(0, len(v_roles)):
        nv = p_frame[k]
        found = False
        for r in range(0,self.nref):
#          print nv
#          print _ref[1 + 2*max_r*r + 2*k+1]
          val = _ref[1 + 2*max_r*r + 2*k + 1]
          if nv != unk_symbol and nv == val :
            found = True
            break
        if not found: all_found = False
        if found and verb_found: _score_card["value"] += 1
        if found: _score_card["value*"] += 1
#        print found
#      print all_found
      if all_found and verb_found: _score_card["value-all"] += 1
      if all_found: _score_card["value-all*"] += 1
  
  def combine(self, rv, card):
    for (k,v) in card.items(): rv[k] += v

  def get_average_results(self, groups = []):
    #average across score cards.  
    rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
    agg_cards = {}
    for (key, card) in self.score_cards.items():
      (v,g) = key
      if len(groups) == 0 or g in groups:
        if v not in agg_cards: 
          new_card = {"verb":0.0, "n_image":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
          agg_cards[v] = new_card
        self.combine(agg_cards[v], card)
    nverbs = len(agg_cards)
    for (v, card) in agg_cards.items():
      img = card["n_image"] 
      nvalue = card["n_value"]
      rv["verb"] += card["verb"]/img
      rv["value-all"] += card["value-all"]/img
      rv["value-all*"] += card["value-all*"]/img
      rv["value"] += card["value"]/nvalue
      rv["value*"] += card["value*"]/nvalue
      
    rv["verb"] /= nverbs
    rv["value-all"] /= nverbs
    rv["value-all*"] /= nverbs 
    rv["value"] /= nverbs
    rv["value*"] /= nverbs

    return rv


 
 
class imSituVerbRoleNounEncoder:
  
  def n_verbs(self): return len(self.v_id)
  def n_nouns(self): return len(self.n_id)
  def n_roles(self): return len(self.r_id)
  def verbposition_role(self,v,i): return self.v_r[v][i]
  def verb_nroles(self, v): return len(self.v_r[v])
  def max_roles(self): return self.mr  
  def pad_symbol(self): return self.n_id["pad"] 
  def unk_symbol(self): return self.n_id["unk"]
  def verb_roles(self,v): return self.v_r[v]  

  def __init__(self, dataset, metadata = "imsitu_space.json",  unk = -1):


    self.v_id = {}
    self.id_v = {}
   
    self.r_id = {}
    self.id_r = {}

    self.id_n = {}
    self.n_id = {}

    self.mr = 0
 
    self.v_r = {} 
    self.noun_freq = {}
  
    self._n_nouns = 0
 
    self.r_id["pad"] = 0
    self.id_r[0] = "pad"
     
    for (image, annotation) in dataset.items():
      v = annotation["verb"]
      if v not in self.v_id: 
        _id = len(self.v_id)
        self.v_id[v] = _id
        self.id_v[_id] = v
        self.v_r[_id]  = []
      vid = self.v_id[v]
      for frame in annotation["frames"]:
        for (r,n) in frame.items():
          if r not in self.r_id: 
            _id = len(self.r_id)
            self.r_id[r] = _id
            self.id_r[_id] = r

          if n not in self.n_id: 
            _id = len(self.n_id)
            self.n_id[n] = _id
            self.id_n[_id] = n
            self.noun_freq[n] = 0
          self.noun_freq[n] +=1 
          rid = self.r_id[r]
          if rid not in self.v_r[vid]: self.v_r[vid].append(rid)                    
   
    #unking
    print ("nouns before unk {}".format(len(self.n_id)))
    self.n_id = {}
    self.id_n = {}
   
    self.n_id["pad"] = 0
    self.id_n[0] = "pad"
     
    #self.n_id["unk"] = 1
    #self.id_n[1] = "unk"
    #make pad and unk symbol identical
    self.n_id["unk"] = 1
    self.id_n[1] = "unk"

    for (n,c) in self.noun_freq.items():
      if c >= unk :
         _id = len(self.n_id)
         self.n_id[n] = _id
         self.id_n[_id] = n
    print ("nouns after unk {}".format(len(self.n_id)))

    for (v,rs) in self.v_r.items(): 
      if len(rs) > self.mr : self.mr = len(rs)
    
    for (v, vid) in self.v_id.items():  self.v_r[vid] = sorted(self.v_r[vid])

    metadata = json.load(open(metadata))["verbs"]
    self.v_order = {}
    for (v, vid) in self.v_id.items():
      _o = {}
      o = metadata[v]["order"]
      i = 0
      for r in o:
        rid = self.r_id[r]
        _o[rid] = i
        i+=1
      self.v_order[vid] = _o
       
    print(len(self.v_order))

  def encode(self, situation):
    rv = {}
    verb = self.v_id[situation["verb"]]
    rv["verb"] = verb
    rv["frames"] = []
    for frame in situation["frames"]:
      _e = []
      for (r,n) in frame.items():
        if r in self.r_id: _rid = self.r_id[r]
        else: _rid = self.pad_symbol() #bug, it doesn't techinically have to be the unk
        if n in self.n_id: _nid = self.n_id[n]
        else: _nid = self.pad_symbol() #bug
        _e.append((_rid, _nid))
      #print(verb)
      #print(self.v_order[verb])
      #print(_e)
      _e = sorted(_e, key=lambda x : self.v_order[verb][x[0]] ) 
      rv["frames"].append(_e)
    boxes = []
    verb = self.v_id[situation["verb"]]
    for (role,box) in situation["bb"].items():  
      if r in self.r_id: _rid = self.r_id[r]
      else: _rid = self.pad_symbol() #bug, it doesn't techinically have to be the unk
      boxes.append((_rid, box))
    _e = sorted(boxes, key=lambda x : self.v_order[verb][x[0]] ) 
    rv["boxes"] = _e
    #print(situation)
    rv["shape"] = [1.0, situation["height"], situation["width"]]
    return rv

  def decode(self, situation):
    verb = self.id_v[situation["verb"]]
    rv = {"verb": verb, "frames":[], "boxes":{}}
    for frame in situation["frames"]:
      _fr = {}
      for (r,n) in frame.items():
        _fr[self.id_r[r]] =  self.id_n[n]
      rv["frames"].append(_fr)
    if "boxes" in situation:
     for (r, box) in situation["boxes"]:
      rv["boxes"][self.id_r[r]] = box
    return rv     

  #takes a list of situations
  def to_tensor(self, situations, use_role = True, use_verb = True):
    rv_semantics = []
    rv_space = []
    for situation in situations:
      _rv = self.encode(situation)
      verb = _rv["verb"]
      items = []
      if use_verb: items.append(verb)
      for frame in _rv["frames"]:
      #sort roles... they are sorted on encode
      #  _f = sorted(frame, key = lambda x : x[0])
        _f = frame
        k = 0
        for (r,n) in _f: 
          if use_role : items.append(r)
          items.append(n)
          k+=1
        while k < self.mr: 
          if use_role: items.append(self.pad_symbol())
          items.append(self.pad_symbol())
          k+=1
      rv_semantics.append(torch.LongTensor(items))
      items = []
      k = 0
      for (r, p) in _rv["boxes"]:
        #print(p)
        items.extend([p[0],p[1], p[2], p[3]])
        if use_role: items.append(r)
      while k < self.mr:
        items.extend([-1, -1, -1, -1])
        if use_role: items.append(self.pad_symbol())
        k+=1
      rv_space.append(torch.FloatTensor(items))
 
    return (torch.cat(rv_semantics), torch.cat(rv_space))
  
  #the tensor is BATCH x VERB X FRAME
  def to_situation(self, tensor_semantics, tensor_space = None):
    (batch,verbd,_) = tensor.size()
    rv = []
    for b in range(0, batch):
      _tensor = tensor_semantics[b]
      #_rv = []
      for verb in range(0, verbd):
        args = []
        __tensor = _tensor[verb]
        for j in range(0, self.verb_nroles(verb)):
          n = __tensor.data[j]
          args.append((self.verbposition_role(verb,j),n))
        
        situation = {"verb": verb, "frames":[args]}
        if tensor_space is not None:
          _tensor = tensor_space[b]
          __tensor = _tensor[verb]
          space = []
          for j in range(0, self.verb_nroles(verb)):
            space.append((self.verbposition_role(verb,j),__tensor.data[j]))
          situation["boxes"] = space
  
        rv.append(self.decode(situation))
          
    return rv

class imSituVerbRoleLocalNounEncoder(imSituVerbRoleNounEncoder):
  
  def n_verbrole(self): return len(self.vr_id)
  def n_verbrolenoun(self): return self.total_vrn
  def verbposition_role(self,v,i): return self.v_vr[v][i]
  def verb_nroles(self, v): return len(self.v_vr[v])
 
  def __init__(self, dataset):
    imSituVerbRoleNounEncoder.__init__(self, dataset)
    self.vr_id = {}
    self.id_vr = {}
   
    self.vr_n_id = {}
    self.vr_id_n = {} 

    self.vr_v = {}
    self.v_vr = {}

    self.total_vrn = 0      

    for (image, annotation) in dataset.items():
      v = self.v_id[annotation["verb"]]
  
      for frame in annotation["frames"]:
        for(r,n) in frame.items(): 
          r = self.r_id[r]
          n = self.n_id[n]

          if (v,r) not in self.vr_id:
            _id = len(self.vr_id)
            self.vr_id[(v,r)] = _id
            self.id_vr[_id] = (v,r)
            self.vr_n_id[_id] = {}
            self.vr_id_n[_id] = {} 
            self.vr_id_n[_id][0] = "pad"
            self.vr_id_n[_id][1] = "unk"
            self.vr_n_id[_id]["pad"] = 0
            self.vr_n_id[_id]["unk"] = 1        

          vr = self.vr_id[(v,r)]    
          if v not in self.v_vr: self.v_vr[v] = []
          self.vr_v[vr] = v
          if vr not in self.v_vr[v]: self.v_vr[v].append(vr)
        
          if n not in self.vr_n_id[vr]:
            _id = len(self.vr_n_id[vr]) 
            self.vr_n_id[vr][n] = _id
            self.vr_id_n[vr][_id] = n
            self.total_vrn += 1

  def encode(self, situation):
    v = self.v_id[situation["verb"]]
    rv = {"verb": v, "frames": []}
    for frame in situation["frames"]:
      _e = [] 
      for (r,n) in frame.items():
        if r not in self.r_id: r = self.unk_symbol()
        else: r = self.r_id[r]
        if n not in self.n_id: n = self.unk_symbol()
        else: n = self.n_id[n]
        if (v,r) not in self.vr_id: vr = self.unk_symbol()
        else: vr = self.vr_id[(v,r)]
        if vr not in self.vr_n_id: vrn = self.unk_symbol()
        elif n not in self.vr_n_id[vr]: vrn = self.unk_symbol()
        else: vrn = self.vr_n_id[vr][n]    
        _e.append((vr, vrn))
      rv["frames"].append(_e) 
    return rv

  def decode(self, situation):
    #print situation
    verb = self.id_v[situation["verb"]]
    rv = {"verb": verb, "frames":[]}
    for frame in situation["frames"]:
      _fr = {}
      for (vr,vrn) in frame:
        if vrn not in self.vr_id_n[vr]: 
          print ("index error, verb = {}".format(verb))
          n = -1
        else:
          n = self.id_n[self.vr_id_n[vr][vrn]]
        r = self.id_r[self.id_vr[vr][1]]
        _fr[r]=n
      rv["frames"].append(_fr)
    return rv 




class imSituSimpleImageFolder(data.Dataset):
 # partially borrowed from ImageFolder dataset, but eliminating the assumption about labels
   def is_image_file(self,filename):
    return any(filename.endswith(extension) for extension in self.ext)  
  
   def get_images(self,dir):
    images = []
    for target in os.listdir(dir):
        f = os.path.join(dir, target)
        if os.path.isdir(f):
            continue
        if self.is_image_file(f):
          images.append(target)
    return images

   def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        #list all images        
        self.ext = [ '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
        self.images = self.get_images(root)
 
   def __getitem__(self, index):
        _id = os.path.join(self.root,self.images[index])
        img = Image.open(_id).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        return img, torch.LongTensor([index])

   def __len__(self):
        return len(self.images)

class imSituSituation(data.Dataset):
   def __init__(self, root, annotation_file, encoder, transform=None, use_space = False, channel_order = "RGB", min_size = 512.0, max_size = 512.0):
        self.root = root
        self.imsitu = annotation_file
        self.ids = list(self.imsitu.keys())
        self.encoder = encoder
        self.transform = transform
        self.use_space = use_space
        self.channel_order = channel_order   
        self.max_size = max_size
        self.min_size = min_size

   def index_image(self, index):
        rv = []
        index = index.view(-1)
        for i in range(index.size()[0]):
          rv.append(self.ids[index[i]])
        return rv
      
   def __getitem__(self, index):
        imsitu = self.imsitu
        _id = self.ids[index]
        ann = self.imsitu[_id]
       
        img = Image.open(os.path.join(self.root, _id)).convert('RGB')
        #if False and max(img.size) > self.max_size:
        #  #scale down
        #  scale = self.max_size / max(img.size)
        #  scale1 = self.max_size / img.size[0]
        #  scale2 = self.max_size / img.size[1]
       #   print("scaled {}".format(scale))
        #  img = img.resize( (int(img.size[0] / scale1), int(img.size[1] / scale2)), resample=PIL.Image.BILINEAR )
          #img = img.resize( (int(img.size[0] / scale), int(img.size[1] / scale)), resample=PIL.Image.BILINEAR )
        #else:
        #  scale = 1

        #real_w = img.size[0]
        #real_h = img.size[1]
        #pad to size
        #right_pad = self.max_size - real_w
        #bottom_pad = self.max_size - real_h
        #print("{} {}".format(real_w, real_h))
        #new_img = Image.new("RGB", (int(self.max_size), int(self.max_size)))
        #new_img.paste(img, (0,0))
        #img = new_img
        #print("{}".format(new_img.size))
        #img = ImageOps.expand(img, (0, 0, right_pad, bottom_pad))
        #print img.size 
        #exit()
        #do some channel switching here potentially, on some flags       
        if not self.channel_order == "RGB": 
          print("only RGB channel order currently supported")
          exit()
 
        if self.transform is not None: img = self.transform(img)
        target_semantics, target_space = self.encoder.to_tensor([ann])

        if self.use_space: return (torch.LongTensor([index]), img, target_semantics, target_space)
        else: return (torch.LongTensor([index]), img, target_semantics)

   def __len__(self):
        return len(self.ids)
