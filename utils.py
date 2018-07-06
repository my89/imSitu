import math
def initLinear(linear, val = None):
  if val is None:
    fan = linear.in_features +  linear.out_features 
    spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
  else:
    spread = val
  print (spread)
  linear.weight.data.uniform_(-spread,spread)
  linear.bias.data.uniform_(-spread,spread)

def initEmbeddings(lookup, word_val = None, word_init = None):
  if word_init is None and word_val is None:
    lookup.weight.data.uniform_(-1, 1)
    return
  if word_val is not None:
    print("{}".format(word_val))
    lookup.weight.data.uniform_(-word_val,word_val)
    return
  print ("fixed word vectors")
  lookup.weight = nn.Parameter(word_init, requires_grad=False)

