import math
def initLinear(linear, val = None):
  if val is None:
    fan = linear.in_features +  linear.out_features 
    spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
  else:
    spread = val
  linear.weight.data.uniform_(-spread,spread)
  linear.bias.data.uniform_(-spread,spread)
