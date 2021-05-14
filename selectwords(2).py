import nltk

from nltk.corpus import stopwords
s=set(stopwords.words('english'))
bow=["the", "stack", "is", "the", "place", "where", "automatically", "allocated" ,"variables", "and", "function", "call",  "return", "addresses", "are", "stored"]
output = []
for w in bow:
    if w not in s:
        output.append(w)
print()
