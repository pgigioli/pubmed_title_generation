class Vocab_Lookup:
    def __init__(self, words, special_tokens=None):
        self.words = words
        self.special_tokens = special_tokens
        self.num_words = None
        self.__word2id_dict = {}
        self.__id2word_dcit = {}

        if self.special_tokens != None:
            for token in self.special_tokens:
                self.__word2id_dict[token] = len(self.__word2id_dict)

        for word in self.words:
            self.__word2id_dict[word] = len(self.__word2id_dict)

        self.__id2word_dict = dict(zip(self.__word2id_dict.values(), self.__word2id_dict.keys()))
        self.num_words = len(self.__word2id_dict)   
    
    def convert_word2id(self, word):
        try:
            word_id = self.__word2id_dict[word]
        except:
            word_id = self.__word2id_dict['UNK']
        return word_id
        
    def convert_id2word(self, word_id):
        return self.__id2word_dict[word_id]
    
    def get_word2id_items(self):
        return self.__word2id_dict.items()
    
    def get_id2word_items(self):
        return self.__id2word_dict.items()