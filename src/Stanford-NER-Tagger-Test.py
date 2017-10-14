from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('C:\\Users\\....\\stanford-ner-2017-06-09\\classifiers\\english.all.3class.distsim.crf.ser.gz',
'C:\\Users\\...\\stanford-ner-2017-06-09\\stanford-ner.jar',
encoding='utf-8')

text = 'your sentence here...'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)