# Realization Templates

This folder contains files supporting realizing situations into simple sentences using templates. 

## generation_templates.tabs

Contains templates for realizing situations. Words preceeding a role name but after the previous role name are removed if the role is either empty or not included in a realization. Role values fill roles in the templates with the first gloss of of the synset from Wordnet.

## realized_parts.tabs.tgz

Contains realized phrases for every subset of parts of realized frames occuring the imSitu training set.

Format: frequency \t verb \t realized_phrase \t subpart
