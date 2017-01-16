import struct
import sys

verb_id = {}
role_id = {}
noun_id = {}
image_id = {}

for line in open("verb_id.tab").readlines():
  tabs = line[0:-1].split("\t")
  verb_id[tabs[1]] = int(tabs[0])

for line in open("role_id.tab").readlines():
  tabs = line[0:-1].split("\t")
  role_id[tabs[1]] = int(tabs[0])

for line in open("noun_id.tab").readlines():
  tabs = line[0:-1].split("\t")
  noun_id[tabs[1]] = int(tabs[0])
noun_id["null"] = -1

for line in open("image_id.tab").readlines():
  tabs = line[0:-1].split("\t")
  image_id[tabs[1]] = int(tabs[0])

def write_int(f , i):
  f.write(struct.pack('i',int(i))) 

infile = open(sys.argv[1])
outfile = open(sys.argv[2],"wb")

for line in infile.readlines(): 
#sys.stdin.readlines():
  tabs = line[0:-1].split("\t") 
  image = image_id[tabs[0]]
  verb = verb_id[tabs[1]]
  write_int(outfile,image)
  write_int(outfile,verb)
  role_noun = []
  for i in range(2,len(tabs),2):
      role = role_id[tabs[i]]
      noun = noun_id[tabs[i+1]]
      #write_int(outfile,role)
      write_int(outfile,noun)
      #role_noun.append((role,noun))
  #outstr = str(image) + "\t" + str(verb)
  #for (k,v) in role_noun:
  #  outstr =outstr+"\t" + str(k) + "\t" + str(v)
  #print outstr

outfile.close()

