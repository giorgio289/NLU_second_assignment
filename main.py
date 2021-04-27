import spacy
import pandas as pd
from conll import evaluate, parse_iob, stats, align_hyp
from spacy.tokens import Doc
from spacy.vocab import Vocab

# class to redefine SpaCy tokenizer
# see https://spacy.io/usage/linguistic-features#tokenization
class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

# get corpus from a CoNLL formatted file
# input: the absolute or relative path of the source file as string
# output: a string made of all tokens from CoNLL separated by a white space
def conll_to_string(path):
    f = open(path, "r")
    words=""
    for line in f:
        if len(line.strip())>0: # check if string is not empty
            spl_line=line.strip().split(' ')
            if spl_line[0]!='-DOCSTART-': #filter -DOCSTART- lines
                words=words+spl_line[0]+" " # add the text to the output string followed by a white space
    f.close()
    return words.rstrip() #remove whitespace from the end of the string before return


# rewrite token entity type in doc to map CoNLL format
# input: a SpaCy document
# output: the same document with updated entity type
def remap(doc):
    for t in doc:
        if t.ent_type_=='PERSON':
            t.ent_type_='PER'
        elif t.ent_type_=='FAC':
            t.ent_type_='LOC'
        elif t.ent_type_=='GPE':
            t.ent_type_='LOC'
        elif t.ent_type_=='EVENT':
            t.ent_type_='MISC'
        elif t.ent_type_=='LAW':
            t.ent_type_='MISC'
        elif t.ent_type_=='NORP':
            t.ent_type_='MISC'
        elif t.ent_type_=='LANGUAGE':
            t.ent_type_='MISC'
        elif t.ent_type_=='WORK_OF_ART':
            t.ent_type_='MISC'
        else:
            t.ent_type_='O'
    return doc

# get a format usabel with conll.py evaluate function
# input: a SpaCy document
# ouput: list of list of tuple
def get_list_from_doc(doc):
    document_list=[]
    for t in doc:
        # check if the entity type is O to adapt format
        if t.ent_type_!='O':
            token_tuple=(t.text,t.ent_iob_+'-'+t.ent_type_) # generate tuple
            document_list.append([token_tuple]) # add tuple a to list of lists
        else:
            token_tuple=(t.text,t.ent_type_) # generate tuple
            document_list.append([token_tuple]) # add tuple a to list of lists
    return document_list

# get a format usabel with conll.py evaluate function
# input: a CoNLL file path
# ouput: list of list of tuple
def get_list_from_conll(path):
    f = open(path, "r")
    document_list=[]
    for line in f:
        if len(line.strip())>0: #check if line is empty
            spl_line=line.strip().split(' ')
            if spl_line[0]!='-DOCSTART-': # filter -DOCSTART- lines
                token_tuple=(spl_line[0],spl_line[-1]) # generate tuple
                document_list.append([token_tuple]) # add tuple to list of lists
    f.close()
    return document_list

# compute accuracy from reference and correct value numbers
def score_tok(cor,ref):
    a = 1 if (cor == 0 and ref == 0) else cor / ref # if both cor and ref are zeros (no token of this kind) set acc to 1 else compute accuracy
    return {"acc": a}

# evaluate token accuracy per class and total
# input: a test as string
# output: results dictionary
def evaluate_token(refs,hyps):
    aligned = align_hyp(refs, hyps)
    tok = stats()
    cls={}
    for sent in aligned:
        for t in sent:
            hyp_iob, hyp = parse_iob(t[-1]) #hyp parsing
            ref_iob, ref = parse_iob(t[-2]) #ref parsing
            if(ref != None): # looks for tagged obj
                if hyp==None: # if untagged set O tag for hypothesis
                    compl_hyp='O'
                else:
                    compl_hyp = hyp_iob + "-" + hyp # get complete NER tag
                compl_ref = ref_iob + "-" + ref

                #initialization of classes
                if not cls.get(compl_ref) and ref:
                    cls[compl_ref] = stats()
                if not cls.get(compl_hyp) and hyp:
                    cls[compl_hyp] = stats()

                if compl_ref == compl_hyp: #check correspondence
                        tok['cor'] += 1
                        cls[compl_ref]['cor'] += 1
                tok['ref'] += 1 #total count for total acc
                cls[compl_ref]['ref'] += 1 #total count for class acc
            else: #do the same in case of O class
                compl_ref='O'
                if hyp==None:
                    compl_hyp='O'
                else:
                    compl_hyp=hyp_iob+"-"+hyp
                if not cls.get(compl_ref):
                    cls[compl_ref] = stats()
                if not cls.get(compl_hyp):
                    cls[compl_hyp] = stats()
                if compl_ref == compl_hyp: #check correspondence
                        tok['cor'] += 1
                        cls[compl_ref]['cor'] += 1
                tok['ref'] += 1 #total count for total acc
                cls[compl_ref]['ref'] += 1 #total count for class acc
    
    results = {lbl: score_tok(cls[lbl]['cor'], cls[lbl]['ref']) for lbl in set(cls.keys())} # store results in a dictionary using score function redacted from conll.py one
    results.update({"total": score_tok(tok.get('cor', 0), tok.get('ref', 0))})
    return results


def group_entities(text):
    def compute_frequency(lista):
        def stats(total):
            return {'rel': 1/total, 'abs': 1}

        res={}
        total=len(lista)
        for sub in lista:
            name=" ".join(sub)
            if name:
                if not res.get(name):
                    res[name]=stats(total)
                else:
                    res[name]['abs']+=1
                    res[name]['rel']=res[name]['abs']/total
        return res

    nlp = spacy.load('en_core_web_sm')
    doc=nlp(text)
    i=0
    out=[]
    while i<len(doc):
        if doc[i].ent_iob_=='B' and doc[i].ent_type_!='':
            found=False
            for chunk in doc.noun_chunks:
                if doc[i] in [word for word in chunk]:
                    tmp=[]
                    found=True
                    for word in chunk:
                        if word.ent_iob_=='B' and word.ent_type_!='':
                            tmp.append(word.ent_type_)
                    out.append(tmp)
                    i=i+len(chunk)
                    break
            if not found:
                out.append([doc[i].ent_type_])
                i=i+1
        else:
            i=i+1
    freq=compute_frequency(out)
    return (out,freq)

def print_freq(dictionary):
    for key,value in dictionary.items():
        print(key)
        for sub_key,freq in value.items():
            print("   {0}: {1}".format(sub_key,freq))

# fix segmentation error using compund relations
# input: a test as string
# output: list of tuples of the form (text,NER_tag)
def fix_segmentation(text):
    nlp = spacy.load('en_core_web_sm')
    doc=nlp(text)
    for t in doc:
        if t.dep_=='compund' and t.head.ent_type_!='': #checks for compounds in tagged tokens
            t.ent_type_=t.head.ent_type_ # set token entity type to his head one
            # set iob tag accordingly to the position of the token with respect to his head
            if t.head.i < t.i: 
                t.ent_iob_='I'
            elif t.head.ent_iob_=='B':
                t.head.ent_iob_='I'
                t.ent_iob_='B'
            else:
                t.ent_iob_='B'            
    out=[]
    # store results in a pretty way
    for t in doc:
        if t.ent_type_=='':
            out.append((t.text,t.ent_iob_))
        else:
            out.append((t.text, t.ent_iob_+"-"+t.ent_type_))
    return out


def evaluate_NER_on_CoNLL(path):
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    # comment the following two lines and uncomment the third to load doc from disk insted of recompute it
    doc = nlp(conll_to_string("./data/test.txt"))
    doc.to_disk("./data/test_out")
    #doc = Doc(Vocab()).from_disk("./data/test_out")
    doc_remapped=remap(doc)
    hyps=get_list_from_doc(doc_remapped)
    refs=get_list_from_conll("./data/test.txt")

    results = evaluate(refs, hyps)
    pd_tbl_chunk = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl_chunk.round(decimals=3)

    results=evaluate_token(refs, hyps)
    pd_tbl_token = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl_token.round(decimals=3)
    return (pd_tbl_chunk,pd_tbl_token)


def test_assignment():
    text = "Apple's Steve Jobs died in 2011 in Palo Alto, California."
    path = "./data/test.txt"
    eval_result = evaluate_NER_on_CoNLL(path)
    ent_result = group_entities(text)
    print("----- BEGIN -----")
    print("Class level performance of SpaCy on " + path)
    print(eval_result[0])
    print("\nToken level accuracy of SpaCy on " + path)
    print(eval_result[1])
    print("\nFixed segmentation error on: "+text)
    print(fix_segmentation(text))
    print("\nGrouped entities of: "+text)
    print(ent_result[0])
    print("\nFrequencies:")
    print_freq(ent_result[1])
    print("-----  END  -----")

test_assignment()