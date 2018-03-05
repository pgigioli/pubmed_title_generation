import re
from vocab import Vocab_Lookup
from sys import stdout
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer()

def load_cnn_dailymail_file(filename):
    story = []
    summary = []
    document = []
    with open(filename, 'r') as read_file:
        start_summary = False
        for line in read_file:
            if line == '\n':
                continue
            document.append(line)
            if start_summary == False:
                if line.find('@highlight') == -1:
                    story.append(line)
                else:
                    start_summary = True
            else:
                if line.find('@highlight') == -1:
                    summary.append(line)
                else:
                    pass
    story = ' '.join([x.rstrip() for x in story])
    summary = '. '.join([x.rstrip() for x in summary])
    return story, summary

def load_pubmed_file(filename):
    with open(filename, 'r') as read_file:
        lines = read_file.readlines()
    title = lines[0].rstrip()
    abstract = lines[1].rstrip()
    return abstract, title

def create_vocab_lookup(files, max_vocab_size=None, special_tokens=None):
    if special_tokens == None:
        special_tokens = ['PAD', 'UNK', 'GO', 'EOS']
        
    word_counter = {}
    err_ct = 0
    ct = 0
    for i, filename in enumerate(files):
        try:
            title, abstract = load_file(filename)
            text = tokenize_sentences(title, tokenizer=toktok) + tokenize_sentences(abstract, tokenizer=toktok)
            ct += 1
            if ct % 1000 == 0:
                stdout.write('\rNumber of documents read: {}'.format(ct))
                stdout.flush()
        except:
            err_ct += 1
            stdout.write('\rNumber of documents unable to decode: {}'.format(err_ct))
            stdout.flush()
            continue
        for word in text:
            word_counter[word] = word_counter.get(word, 0) + 1
    stdout.write("\n")
            
    if max_vocab_size == None:
        words = set(word_counter.keys())
    else:
        words = set(sorted(word_counter, key=word_counter.get, reverse=True)[:max_vocab_size-len(special_tokens)])

    return Vocab_Lookup(words, special_tokens=special_tokens)    

def pad_sequence(ids, vocab_lookup, pad_len=None, go=False, eos=False):
    if go == True:
        ids = [vocab_lookup.convert_word2id('GO')] + ids
    
    if pad_len == None:
        if eos:
            ids += [vocab_lookup.convert_word2id('EOS')]
            real_len = len(ids)
        return ids, real_len
    else:
        if len(ids) < pad_len:
            if eos:
                real_len = len(ids) + 1
                ids += [vocab_lookup.convert_word2id('EOS')] + [vocab_lookup.convert_word2id('PAD') for i in range(pad_len - len(ids) - 1)]
            else:
                real_len = len(ids)
                ids += [vocab_lookup.convert_word2id('PAD') for i in range(pad_len - len(ids))]
        elif len(ids) > pad_len:
            real_len = pad_len
            if eos:
                ids = ids[:pad_len-1] + [vocab_lookup.convert_word2id('EOS')]
            else:
                ids = ids[:pad_len]
        else:
            real_len = pad_len
            if eos:
                ids = ids[:-1] + [vocab_lookup.convert_word2id('EOS')]
            else: 
                ids = ids
        return ids, real_len
    
def extract_sentences(s):
    s = re.sub(' +', ' ', s)
    indices = [m.span()[0] for m in re.finditer(r'[.]', s)]
    
    new_indices = [0]
    for idx in indices:
        if idx != 0 and idx+1 != len(s) and idx+2 != len(s):
            if s[idx-1].isalpha() and s[idx+1] == ' ' and s[idx+2].isupper():
                new_indices.append(idx+1)  
    return [s[i:j].strip() for i,j in zip(new_indices, new_indices[1:]+[None])]

def clean_string(s, lower=True):
    if lower == True:
        return re.sub('[^A-Za-z0-9. ]+', " ", s).lower()
    else:
        return re.sub('[^A-Za-z0-9. ]+', " ", s)
    
def tokenize_sentence(s, tokenizer=None):
    if tokenizer == None:
        return s.replace('.', ' . ').strip().split()
    else:
        return tokenizer.tokenize(s)

def tokenize_sentences(s, tokenizer=None):
    return [word for sent in [tokenize_sentence(clean_string(sent), tokenizer=tokenizer) 
                   for sent in extract_sentences(s)] for word in sent]

def convert_text_to_ids(text, vocab_lookup, pad_len=None, go=False, eos=False):
    text = tokenize_sentences(text, tokenizer=toktok)
    
    ids = [vocab_lookup.convert_word2id(word) for word in text]
    ids, real_len = pad_sequence(ids, vocab_lookup, pad_len=pad_len, go=go, eos=eos)
    return ids, real_len
    
def create_examples_from_file(filenames, source_len, target_len, vocab_lookup, dataset, use_go=False, use_eos=True, include_oov=False, token='word'):
    examples = []
    for filename in filenames:
        if dataset == 'pubmed':
            source, target = load_pubmed_file(filename)
        elif dataset == 'cnn_dailymail':
            source, target = load_cnn_dailymail_file(filename)
        examples.append(Example(source, target, source_len, target_len, vocab_lookup, include_oov=include_oov, token=token))
    return examples
 
def get_oov_words(words, vocab_lookup):
    oov_words = []
    for word in words:
        if vocab_lookup.convert_word2id(word) == vocab_lookup.convert_word2id('UNK'):
            oov_words.append(word)
    return list(set(oov_words))

class Example:
    def __init__(self, source, target, source_len, target_len, vocab_lookup, include_oov=False, token='word'):
        self.source_text = source
        self.target_text = target
        
        if token == 'char':
            self.source_ids, self.source_len = pad_sequence([vocab_lookup.convert_word2id(char) for char in self.source_text], 
                                                            vocab_lookup, pad_len=source_len, eos=True)
            self.target_ids, self.target_len = pad_sequence([vocab_lookup.convert_word2id(char) for char in self.target_text],
                                                            vocab_lookup, pad_len=target_len, eos=True)
        else:
            self.source_ids, self.source_len = convert_text_to_ids(source, vocab_lookup, pad_len=source_len, eos=True)
            self.target_ids, self.target_len = convert_text_to_ids(target, vocab_lookup, pad_len=target_len, eos=True)

            if include_oov == True:    
                self.source_extended_ids = []
                self.target_extended_ids = []

                self.source_words = tokenize_sentences(source, tokenizer=toktok)
                self.target_words = tokenize_sentences(target, tokenizer=toktok)
                self.oov_words = get_oov_words(self.source_words, vocab_lookup)

                for word in self.source_words:
                    if word in self.oov_words:
                        self.source_extended_ids.append(vocab_lookup.num_words + self.source_words.index(word))
                    else:
                        self.source_extended_ids.append(vocab_lookup.convert_word2id(word))
                self.source_extended_ids, _ = pad_sequence(self.source_extended_ids, vocab_lookup, pad_len=source_len, eos=True)

                for word in self.target_words:
                    if word in self.oov_words:
                        self.target_extended_ids.append(vocab_lookup.num_words + self.source_words.index(word))
                    else:
                        self.target_extended_ids.append(vocab_lookup.convert_word2id(word))
                self.target_extended_ids, _ = pad_sequence(self.target_extended_ids, vocab_lookup, pad_len=target_len, eos=True)