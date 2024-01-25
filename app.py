
from transformers import pipeline
classifier = pipeline("token-classification", model = "samrawal/bert-base-uncased_clinical-ner") 
def getEntities(classifier, input_text): 
    lst = classifier(input_text)
    # Combine words into a single string
    if len(lst)!=0:
        combined_words = []
        for i in range(len(out)):
            if lst[i]['tag'] == 'B-problem':
                combined_words.append(lst[i]['entity'])
            elif lst[i]['tag'] == 'I-problem':
                combined_words[-1] += ' ' + lst[i]['entity']
        
        # Create dictionary with B-problem and corresponding I-problem tags, words, and probabilities
        d = {}
        for i in range(len(lst)):
            if lst[i]['tag'] == 'B-problem':
                d[combined_words.index(lst[i]['entity'])] = {'entity': combined_words[combined_words.index(lst[i]['entity'])],
                                                              'tag': lst[i]['tag'],
                                                              'probability': lst[i]['probability']}
            elif lst[i]['tag'] == 'I-problem':
                d[combined_words.index(lst[i-1]['entity'])]['entity'] = combined_words[combined_words.index(lst[i-1]['entity'])]
                d[combined_words.index(lst[i-1]['entity'])]['entity'] += ' ' + lst[i]['entity']
                d[combined_words.index(lst[i-1]['entity'])]['probability'] = lst[i]['probability']
        
        # Print the dictionary
        print(d)
        return d
    else:
        d = "No entities found for your text"
        return d
input_text = input("enter user ED Doc")
output = getEntities(classifier, input_text)
print(input_text," ====== ","\n", "\nOUTPUT\n",output)
