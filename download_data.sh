#!/bin/bash


cd data

# Data for word2vec

if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi


# Data for GPT-1
# Already in the git, downlowned from 
# https://docs.google.com/spreadsheets/d/1FkdPMd7ZEw_Z38AsFSTzgXeiJoLdLyXY_0B_0JIJIbw/edit#gid=81257118
# https://docs.google.com/spreadsheets/d/11tfmMQeifqP-Elh74gi2NELp0rx9JMMjnQ_oyGKqCEg/edit#gid=410941117

cd -