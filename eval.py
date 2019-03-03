import argparse
import imp
import os
import torch
import json
from imsitu import imSituVerbRoleLocalNounEncoder
from imsitu import imSituVerbRoleNounEncoder
from imsitu import imSituTensorEvaluation 
from imsitu import imSituSituation

def format_dict(d, s, p):
    rv = ""
    for (k,v) in d.items():
      if len(rv) > 0: rv += " , "
      rv+=p+str(k) + ":" + s.format(v*100)
    return rv

# essentially we doing this to validate the model's encoding isn't 
# making the problem easier by accident
#
# its a bit slow, becaue you need to convert one encoding to another. 
# idealy, if one uses this code base, and change the encoding,
# they can validate once, and then just make sure it works with imSituTensorEvaluation 
def eval_model(dataset, dataset_loader, standard_encoding, model_encoding, model, trustedEncoder = False, image_group = {}):
    model.eval()
    print ("evaluating model...")
    if trustedEncoder == False:
      print ("not using trusted encoder. This may take signficantly longer as predictions are converted to other encoding.")
    mx = len(dataset_loader) 
    batches = []
    top1 = imSituTensorEvaluation(1, 3, image_group)
    top5 = imSituTensorEvaluation(5, 3, image_group)
    for i, (indexes, input, target) in enumerate(dataset_loader):
      if True or i % 10 == 0: print ("batch {} out of {}\r".format(i+1,mx)),
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      #target_var = torch.autograd.Variable(target.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      (s_sorted, idx) = torch.sort(scores, 1, True)
      if not trustedEncoder:
        predictions = standard_encoding.to_tensor(model_encoding.to_situation(predictions), False, False)
        predictions = predictions.view(target.size()[0], standard_encoding.n_verbs(), -1)  
      else:
        predictions = predictions.data
      #(s_sorted, idx) = torch.sort(scores, 1, True)
      top1.add_point(target, predictions, idx.data, dataset.index_image(indexes))
      top5.add_point(target, predictions, idx.data, dataset.index_image(indexes))
    return (top1, top5) 

#assumes the predictions are grouped by image, and sorted 
def eval_file(output, dataset, standard_encoding, image_group):
  
  f = open(output)
  
  top1 = imSituTensorEvaluation(1, 3, image_group)
  top5 = imSituTensorEvaluation(5, 3, image_group)
  curr_image = ""
  predictions = []
  #indexes = torch.LongTensor(range(0,standard_encoding.n_verbs())).view(1,-1)
  order = []
  j = 0
  for line in f:
    tabs = line.split("\t")
    if curr_image == "": curr_image = tabs[0]
    elif curr_image != tabs[0]:
      #print dataset[curr_image]
      predictions = sorted(predictions, key=lambda x: standard_encoding.v_id[x["verb"]])
      #print len(predictions)
      encoded_predictions = standard_encoding.to_tensor(predictions, False, False).view(1,standard_encoding.n_verbs(), -1)   
      #print encoded_predictions #== -2
      indexes = torch.LongTensor(order).view(1, -1)
      encoded_reference = standard_encoding.to_tensor([dataset[curr_image]]).view(1,-1)
      top1.add_point(encoded_reference, encoded_predictions, indexes, [curr_image])
      top5.add_point(encoded_reference, encoded_predictions, indexes, [curr_image])
      curr_image =tabs[0]
      predictions = []
      order = []
      print ("batch {} out of {}\r".format(j,len(dataset))),
      #if j == 1000: return (top1,top5)
      j+=1
    image_id = tabs[0]
    verb = tabs[1].strip()
    roles = {}
    #print tabs
    for i in range(2, len(tabs), 2):
      if tabs[i+1].strip() == "null": n = ""
      else: n = tabs[i+1].strip()
      roles[tabs[i].strip()] = n#tabs[i+1]
    predictions.append({"verb":verb,"frames":[roles]})
    #print predictions 
    order.append(standard_encoding.v_id[verb])
  #last one
  predictions = sorted(predictions, key=lambda x: standard_encoding.v_id[x["verb"]])
  encoded_predictions = standard_encoding.to_tensor(predictions, False, False).view(1,standard_encoding.n_verbs(), -1)    
  indexes = torch.LongTensor(order).view(1, -1)
  encoded_reference = standard_encoding.to_tensor([dataset[curr_image]]).view(1,-1)
  top1.add_point(encoded_reference, encoded_predictions, indexes, [curr_image])
  top5.add_point(encoded_reference, encoded_predictions, indexes, [curr_image])
  return (top1,top5)
      

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="imsitu independant evaluation script.") 
  parser.add_argument("--format", choices=["file", "model"], required=True)
  parser.add_argument("--weights_file", help="the model to start from")
  parser.add_argument("--encoding_file", help="a file corresponding to the encoder")
  parser.add_argument("--system_output", help="file to read predictions from")
  parser.add_argument("--include", help="name of file in model folder to include")
  parser.add_argument("--sparsity_min", default = 0, type=int, help="only evaluate images that meet some sparsity threshold")
  parser.add_argument("--sparsity_max", default = -1, type=int, help="only evaluate images that meet some sparsity threshold")
  parser.add_argument("--eval_file", default="dev.json", help="evaluation file")
  parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
  parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
  parser.add_argument("--dataset_dir", default="./", help="location of train.json, dev.json, ect.") 
  parser.add_argument("--trust_encoder", default=False, help="if trusted, skip re-encoding intro trusted, which takes a long time", action='store_true')
 
  args = parser.parse_args()
  if args.sparsity_min > (args.sparsity_max+1):
    print ("sparsity_min must be less than or equal to sparsity_max")
    exit()
  train_set = json.load(open(args.dataset_dir+"/train.json"))
  #compute sparsity statistics 
  verb_role_noun_freq = {}
  for image,frames in train_set.items():
    v = frames["verb"]
    items = set()
    for frame in frames["frames"]: 
      for (r,n) in frame.items():
        key = (v,r,n)
        items.add(key)
    for key in items:
      if key not in verb_role_noun_freq: verb_role_noun_freq[key] = 0
      verb_role_noun_freq[key] += 1
   #per role it is the most frequent prediction
   #and among roles its the most rare
  
  eval_dataset = json.load(open(args.dataset_dir + "/" + args.eval_file))  
  image_sparsity = {}
  for image,frames in eval_dataset.items():
     v = frames["verb"]
     role_max = {}
     for frame in frames["frames"]:
       for (r,n) in frame.items():
         key = (v,r,n)
         if key not in verb_role_noun_freq: freq = 0
         else: freq = verb_role_noun_freq[key] 
         if r not in role_max or role_max[r] < freq: role_max[r] = freq    
     min_val = -1
     for (r,f) in role_max.items(): 
       if min_val == -1 or f < min_val: min_val = f 
     image_sparsity[image] = min_val
  if args.sparsity_max > -1:
    x = range(args.sparsity_min, args.sparsity_max+1)
    print ("evaluating images where most rare verb-role-noun in training is x , s.t. {} <= x <= {}".format(args.sparsity_min, args.sparsity_max))
    n = 0
    for (k,v) in image_sparsity.items():
      if v in x: n+=1
    print ("total images = {}".format(n))

  if args.format == "model":
    standard_encoder = imSituVerbRoleNounEncoder(train_set)
    mod_name,file_ext = os.path.splitext(os.path.split(args.include)[-1])
    model_module = imp.load_source(mod_name, args.include)
    
    if args.encoding_file is None: 
      print ("expecting encoder file to run evaluation")
      exit()
    else:
      encoder = torch.load(args.encoding_file)
    print ("creating model...") 
    model = getattr(model_module, mod_name)(encoder)
    if args.weights_file is None:
      print ("expecting weight file to run features")
      exit()
    
    print ("loading model weights...")
    model.load_state_dict(torch.load(args.weights_file))
    model.cuda()

    dataset_encoder = standard_encoder
    if args.trust_encoder: dataset_encoder = encoder   
 
    dataset = imSituSituation(args.image_dir, eval_dataset, dataset_encoder, model.dev_preprocess())
    loader  = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = 3) 
    
    (top1, top5) = eval_model(dataset, loader, standard_encoder, encoder, model, args.trust_encoder, image_sparsity)    
  
  elif args.format == "file":
    standard_encoder = imSituVerbRoleNounEncoder(train_set)
    if args.system_output is None:
      print ("expecting output file")
      exit()
    (top1, top5) = eval_file(args.system_output, eval_dataset, standard_encoder, image_sparsity)
 
   
  top1_a = top1.get_average_results(range(args.sparsity_min, args.sparsity_max+1))
  top5_a = top5.get_average_results(range(args.sparsity_min, args.sparsity_max+1))

  avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
  avg_score /= 8

  print ("\ntop-1\n\tverb     \t{:.2f}%\n\tvalue    \t{:.2f}%\n\tvalue-all\t{:.2f}%\ntop-5\n\tverb     \t{:.2f}%\n\tvalue    \t{:.2f}%\n\tvalue-all\t{:.2f}%\ngold verbs\n\tvalue    \t{:.2f}%\n\tvalue-all\t{:.2f}%\nsummary \n\tmean    \t{:.2f}%".format(100*top1_a["verb"], 100*top1_a["value"], 100*top1_a["value-all"], 100*top5_a["verb"], 100*top5_a["value"], 100*top5_a["value-all"], 100*top5_a["value*"], 100*top5_a["value-all*"], 100*avg_score))


  #print "Average :{:.2f} {} {}".format(avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-"))    
