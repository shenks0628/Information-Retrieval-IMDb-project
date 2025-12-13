import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, additional_stop_words=None):
        self._download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        if additional_stop_words:
            self.stop_words.update(additional_stop_words)

    def _download_nltk_resources(self):
        """檢查並下載必要的 NLTK 資源"""
        resources = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng']
        for res in resources:
            try:
                print(f"Checking NLTK resource: {res}")
                nltk.data.find(f'tokenizers/{res}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{res}')
                except LookupError:
                    print(f"Downloading NLTK resource: {res}")
                    nltk.download(res, quiet=True)

    def _get_wordnet_pos(self, tag):
        """將 Treebank POS tag 轉換為 WordNet POS tag"""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def process(self, text):
        """對單一文本進行斷詞、標註詞性與詞形還原"""
        tokens = word_tokenize(text.lower())
        tags = pos_tag(tokens)
        lemmas = []
        for tag in tags:
            # 只保留字母且不在停用詞表中的詞
            if tag[0].isalpha() and tag[0] not in self.stop_words:
                wordnet_pos = self._get_wordnet_pos(tag[1]) or wordnet.NOUN
                lemmas.append(self.lemmatizer.lemmatize(tag[0], pos=wordnet_pos))
        return ' '.join(lemmas)