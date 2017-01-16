# imSitu
Suport, annotation, and evaluation files for the imSitu dataset. 

If you use the imSitu dataset in your research, please cite:

@inproceedings{yatskar2016,
  title={Situation Recognition: Visual Semantic Role Labeling for Image Understanding},
  author={Yatskar, Mark and Zettlemoyer, Luke and Farhadi, Ali},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2016}
}

## imsitu_space.json
A json file defining the metadata used for imSitu. 

Each verb in imSitu is mapped to a FrameNet frame, for example, 'clinging' is mapped to Retaining, and each of its roles is mapped to a role associated with that frame, for example, the imSitu role 'clungto' for clinging is mapped to the framenet role 'theme'. Each noun in space corresponds to a synset from Wordnet.
 
imsitu_space.json contians these mappings, as well as definitions of verbs, roles and an abstract sentence presented to turkers to help them understand the roles quickly. Furthermore, it contains definitions of nouns and glosses for those nouns that were presented to turkers.
 
```
import json

imsitu = json.load(open("imsitu_space.json"))

nouns = imsitu["nouns"]
verbs = imsitu["verbs"]

verbs["clinging"]

# {u'abstract': u'an AGENT clings to the CLUNGTO at a PLACE',
#  u'def': u'stick to',
#  u'framenet': u'Retaining',
#  u'order': [u'agent', u'clungto', u'place'],
#  u'roles': {
#   u'agent': {u'def': u'The entity doing the cling action',u'framenet': u'agent'},
#   u'clungto': {u'def': u'The entity the AGENT is clinging to',u'framenet': u'theme'},
#   u'place': {u'def': u'The location where the cling event is happening',u'framenet': u'place'}
#  }
# }

nouns["n02129165"]

#{u'def': u'large gregarious predatory feline of Africa and India having a tawny coat with a shaggy mane in the male',
# u'gloss': [u'lion', u'king of beasts', u'Panthera leo']}

```
## train.json, dev.json, test.json

These files contain annotations for the train/dev/test set. Each image in the imSitu dataset is annotated with three frames corresponding to one verb. Each annotation contains a noun value from Wordnet, or the empty string, indicating empty, for every role associated with the verb

```
import json
train = json.load(open("train.json"))

train['clinging_250.jpg']
#{u'frames': [{u'agent': u'n01882714', u'clungto': u'n05563770', u'place': u''},
#  {u'agent': u'n01882714', u'clungto': u'n05563770', u'place': u''},
#  {u'agent': u'n01882714', u'clungto': u'n00007846', u'place': u''}],
# u'verb': u'clinging'}

```

## images
Images resized to 256x256 here (3.7G):

https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar

Original images can be found here (34G) :

https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar

## evaluation
See the readme in evaluation directory
