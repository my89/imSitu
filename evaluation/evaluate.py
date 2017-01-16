import sys
import json
import argparse

parser = argparse.ArgumentParser("Score situation recognition output")
parser.add_argument("--sys_output", help="system output", required=True)
parser.add_argument("--input_type", help="human|ids", choices=["human", "ids"], default="human")
parser.add_argument("--ref_set", help="the reference set to compare against", default="dev.json")
parser.add_argument("--sparsity_min_max", help="min and max threshold ( for example, 0,0 reports performance for images with role values that have no training examples)", nargs=2, required=False, type=int)
parser.add_argument("--histogram", help="meausre to output for histograph. Histogram is done by frequency (max over references min over roles) of verb-role-noun combinatins in the training set", required=False)
parser.add_argument("--topk", help="the topk outputs to evaluate. For example, 1 5 will evaluate 1 and 5", nargs="+", default=[1,5], type=int)
parser.add_argument("--train_set", help="train set file for computing verb-role-noun frequencies" , default="train.json")
args = vars(parser.parse_args())
#input a prediction for every verb
sys_set = open(args["sys_output"]) #open(sys.argv[1])
ref_set = open(args["ref_set"])
print(args)
if args["sparsity_min_max"] is None:
  zero_shot_only = False
  sparsity_max = -1
  sparsity_min = -1
else:
  zero_shot_only = True
  sparsity_min = args["sparsity_min_max"][0]
  sparsity_max = args["sparsity_min_max"][1]  
if args["histogram"] is not None:
  histogram = 1
  do_hist = True
  if not len(args["topk"]) == 1: 
    print "histogram only supported when --topk has exactly one integer"
    exit()
else:
  histogram = 0
  do_hist = False

train_set = open(args["train_set"])
train = json.load(train_set) 

if zero_shot_only :
  print "applying sparsity threshold >= {0} and <= {1}".format(sparsity_min, sparsity_max)

eval_points = args["topk"]
max_eval = max(eval_points)

ref = json.load(ref_set);

#go through and find the subset that is never seen 

structure = {}
#go through and get the structure def so that we can inform if the sys structure is messed up
for (image,s) in ref.items():
	if s["verb"] not in structure:
		roles = set()
		for (r, n) in s["frames"][0].items():
			roles.add(r)
		structure[s["verb"]] = roles
if zero_shot_only or do_hist:
  hist = {}
  images = set()
  freq = {}
  for (k,v) in train.items():
    verb = v["verb"]
    _freq = {}
    for frame in v["frames"]:
      for (r,n) in frame.items():
        key = verb + "_" + r + "_" + n
        if key not in _freq: _freq[key] = 0
        _freq[key] += 1
    for (key, cnt) in _freq.items():
      if key not in freq: freq[key] = 0
      freq[key] += 1
 
  for (img,v) in ref.items():
    verb = v["verb"]
    role_freq = {}
    for frame in v["frames"]:
      for (r,n) in frame.items():
        key = verb + "_" + r + "_" + n
        if key not in freq: f = 0
        else: f = freq[key]
        if r not in role_freq: role_freq[r] = f
        else: role_freq[r] = max(f, role_freq[r])
    sorted_items = sorted(role_freq.items(), key = lambda x: x[1])
    #for (k,f) in role_freq.items():
    f = sorted_items[0][1]
    if zero_shot_only and f <= sparsity_max and f >= sparsity_min :
      #its zero shot 
      images.add(img)
      #break
      if f not in hist: hist[f] = set()
      hist[f].add(img)
    elif do_hist:
      if f not in hist: hist[f] = set()
      hist[f].add(img)

if not do_hist:
  hist = {} 
  _images = set()
  for (img,v) in ref.items(): _images.add(img)
  hist[0] = _images

if zero_shot_only:
  print "total examples = {0} where verb-role-noun occurance <= {1} and >= {2} ".format(len(images), sparsity_max, sparsity_min)

sys = {}
n = 0

#expecting image role noun role noun role noun
for line in sys_set.readlines():
	tabs = line.split("\t")
	image = tabs[0]
	verb = tabs[1]

	if image not in sys: 
		sys[image] = {}

	if image not in ref:
		print "Error on line " + str(n) + " : " + image + " not in reference set "
		exit()
	pred = {}
        if len(sys[image]) >= max_eval and verb != ref[image]["verb"] : 
		n+=1
		continue
	for i in range(0, (len(tabs)-2)/2):
		role = tabs[2+i*2].lower().strip()
		noun = tabs[3+i*2].lower().strip()
		if noun == "null" : noun = ""
		if role not in structure[verb]:
			print "Error on line " + str(n) + " : " + role + " not in role set for that verb ( " + str(structure[verb]) + " )"
			exit()
		pred[role] = noun
	#pred["verb"] = verb.lower().strip()					

	sys[image][verb] = {"n" : n%len(structure) , "p" : pred}
	n+=1		
        print("\rreading input ... {0}/{1} ({2:.2f}%)".format(n,len(ref)*len(structure), (float(n*100))/(len(ref)*len(structure)))),
print n

if len(sys) != len(ref):
	print "Error, missing predictions for images in the reference set:" 
	for image in ref.keys():
		if image not in sys:
			print "  " + str(image)
	exit()

tot = 0
hist_points = sorted(hist.keys())
if len(hist_points) > 1: 
  print "-- frequency histogram --"
  print "n \t %images <= n  \t % @ n"

if len(hist_points) > 1:
  for t in hist_points:
    hist_images = hist[t]
    tot += len(hist_images)
    print str(t)  + "\t" + str(float(tot)/25200*100) + "\t" + str(len(hist_images)/25200.0*100) 

#print hist[231]

all_cards = []
for t in hist_points:
  hist_images = hist[t]
  for k in eval_points:
	verb_cards = {}
	for (image, structures) in sys.items():
                if zero_shot_only and image not in images: continue
                if image not in hist_images: continue
		references = ref[image]
		ref_verb = references["verb"]

		if ref_verb not in verb_cards:
			score_card = {"verb" : 0, "n_image" : 0, "role":0 , "role*":0 , "n_value" :0 , "any" : 0, "any*":0}
			verb_cards[ref_verb] = score_card
		score_card = verb_cards[ref_verb]

		verb_match = False
		score_card["n_image"] += 1
		for (verb, prediction) in structures.items():
			if verb == ref_verb and prediction["n"] < k: 
				score_card["verb"] += 1
				verb_match = True
				break
		
		roles = structures[ref_verb]["p"]
		all_matched = True
		#check if each role matches a partciular reference	
		for (role, noun) in roles.items():
			matched = False
			for an in references["frames"]:
				if an[role] == noun: 
					if verb_match: score_card["role"]+=1
					score_card["role*"] += 1
					matched = True
					break
			all_matched = all_matched and matched
			score_card["n_value"] += 1
		if all_matched:
			if verb_match: score_card["any"] += 1
			score_card["any*"] += 1

	all_cards.append(verb_cards)

#perform an across verb average
tot_score = 0
tot_n = 0
measure_hist = {"verb":[], "role":[], "any":[], "role*":[], "any*":[], "mean":[]}
for i in range(0, len(all_cards)):
	score_card = {"verb" : 0, "role":0 , "role*":0 , "any" : 0, "any*":0}
	verb_cards = all_cards[i]
	for (verb, _score) in verb_cards.items():
		n_image = _score["n_image"] + 0.0
		n_role = _score["n_value"] + 0.0
		score_card["verb"] += _score["verb"]/n_image
		score_card["role"] += _score["role"]/n_role
		score_card["role*"] += _score["role*"]/n_role
		score_card["any"] += _score["any"]/n_image
		score_card["any*"] += _score["any*"]/n_image
		
	n_verbs = len(verb_cards) + 0.0
        if not do_hist:	
	  print "top-" + str(eval_points[i])
	  print "\tverb      \t{0:.2f}%".format(100*score_card["verb"]/n_verbs) 		
	  print "\tvalue     \t{0:.2f}%".format(100*score_card["role"]/n_verbs)  		
	  print "\tvalue-all \t{0:.2f}%".format(100*score_card["any"]/n_verbs)		
          tot_score += 100*(score_card["verb"] + score_card["role"] + score_card["any"])/n_verbs;
          tot_n += 3
	  if i == len(all_cards)-1:
		print "gold verbs"
		print "\tvalue     \t{0:.2f}%".format(100*score_card["role*"]/n_verbs)
		print "\tvalue-all \t{0:.2f}%".format(100*score_card["any*"]/n_verbs)
                tot_score += 100*(score_card["role*"] + score_card["any*"])/n_verbs
                tot_n += 2
          
        else:
          _verb = 100*score_card["verb"]/n_verbs
          _role = 100*score_card["role"]/n_verbs
          _any = 100*score_card["any"]/n_verbs
          _role_s = 100*score_card["role*"]/n_verbs
          _any_s = 100*score_card["any*"]/n_verbs
          _mean = _verb + _role + _any + _role_s + _any_s
          _mean = _mean / 5 
          measure_hist["verb"].append(_verb)
          measure_hist["role"].append(_role)
          measure_hist["role*"].append(_role_s)
          measure_hist["any"].append(_any)
          measure_hist["any*"].append(_any_s)
          measure_hist["mean"].append(_mean)

if not do_hist:
  print "summary\n\tmean      \t{0:.2f}%".format(tot_score/tot_n)
else:
  for (k,v) in measure_hist.items():
      print "-- {0} histogram --".format(k)
      print "n \t {0} score @ n ".format(k) #\t {0} score <= n".format(k) 
      tot = 0
      cum = 0
      for i in range(0,len(v)):
        cur_tot = len(hist[hist_points[i]])
        if not tot == 0:
            cum = (cum * tot + hist_points[i] * cur_tot) / (tot + cur_tot)
        else: 
            cum = v[i]
	print "{0}\t{1}".format(hist_points[i],v[i]) 
        tot += cur_tot 
      print "\n\n"	








