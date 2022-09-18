from math import log2
from django.http import HttpResponse
from django.shortcuts import render
from scipy.sparse import csr_matrix


def home(request):
    return render(request, 'Home.html')

def main_home(request):
    return render(request, 'Main_Home.html')

def func(request):
    djtext = request.GET.get('text', 'default')
    tfidf = request.GET.get('weighting', 'on')
    bigram = request.GET.get('weighting', 'on')
    le = request.GET.get('weighting', 'on')
    rawf = request.GET.get('weighting', 'on')
    rf = request.GET.get('weighting', 'on')
    bf = request.GET.get('weighting', 'on')
    acc = request.GET.get('weighting', 'on')
    acc2 = request.GET.get('weighting', 'on')
    ndm = request.GET.get('weighting', 'on')
    odr = request.GET.get('weighting', 'on')
    gini = request.GET.get('weighting', 'on')


    if tfidf == "tfidf":
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
                 آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
                 ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
                 اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
                 بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
                 تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
                 جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
                 جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
                 دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
                 رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
                 سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
                 فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
                 لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
                 مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
                 نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
                 وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
                 چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
                 کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
                 کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
                 گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
                 ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
                """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')
        from sklearn.feature_extraction.text import TfidfVectorizer
        def token(data):
            doc = nlp(data)
            for i, sentence in enumerate(doc.sentences):
                a = [f"{token.text}" for token in sentence.tokens]
            return a
        vectorizer = TfidfVectorizer(tokenizer=token)
        X = vectorizer.fit_transform([x])
        y = vectorizer.get_feature_names()

        ndarray = X.toarray()
        lst = ndarray.tolist()
        for i in lst:
            import pandas as pd
            dis = {'Weights': i, 'words': y}
            df = pd.DataFrame(dis)
        params = {'purpose': 'TFIDF Weights', 'data': df}
        return render(request, 'tfidf.html', params)

    elif(bigram == "tfidfbigram"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
                 آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
                 ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
                 اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
                 بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
                 تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
                 جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
                 جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
                 دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
                 رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
                 سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
                 فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
                 لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
                 مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
                 نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
                 وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
                 چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
                 کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
                 کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
                 گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
                 ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
                """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')
        from sklearn.feature_extraction.text import TfidfVectorizer
        def token(data):
            doc = nlp(data)
            for i, sentence in enumerate(doc.sentences):
                a = [f"{token.text}" for token in sentence.tokens]
            return a
        vectorizer = TfidfVectorizer(tokenizer=token, ngram_range=(2, 2))
        X = vectorizer.fit_transform([x])
        y = vectorizer.get_feature_names()
        ndarray = X.toarray()
        lst = ndarray.tolist()
        for i in lst:
            import pandas as pd
            dis = {'Weights': i, 'words': y}
            df = pd.DataFrame(dis)
        params = {'purpose': 'TFIDF_bigram Weights', 'data':df}
        return render(request, 'tfidf_bigram.html', params)

    elif(le == "le"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
                 آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
                 ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
                 اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
                 بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
                 تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
                 جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
                 جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
                 دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
                 رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
                 سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
                 فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
                 لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
                 مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
                 نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
                 وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
                 چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
                 کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
                 کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
                 گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
                 ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
                """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text

        data = tokenization([x])
        unique_words = []
        for row in data:
            for word in row:
                if len(word) >= 2 and word not in unique_words:
                    unique_words.append(word)
        unique_words.sort()
        vocab = {j: i for i, j in enumerate(unique_words)}

        from scipy.sparse import csr_matrix
        from collections import Counter
        from math import log2
        FW = []
        Prob = []
        LW = []
        sumf = []
        GW = []
        rows = []
        columns = []
        words = []
        for idx, row in enumerate(data):  # for each document in the dataset
            # it will return a dict type object where key is the word and values is its frequency, {word:frequency}
            word_freq = dict(Counter(row))
            # for every unique word in the document
            for word, freq in word_freq.items():
                words.append(word)
                if len(word) < 2:
                    continue
                # we will check if its there in the vocabulary that we build in fit() function
                # dict.get() function will return the values, if the key doesn't exits it will return -1
                col_index = vocab.get(word, -1)  # retreving the dimension number of a word
                if col_index != -1:  # we are storing the index of the document
                    rows.append(idx)
                    # we are storing the dimensions of the word
                    columns.append(col_index)

                    # Compute Log Entropy
                    lw = log2(freq + 1)
                    LW.append(lw)

                    sfreq = freq
                    sumf.append(sfreq)
                    sm = sum(sumf)  # sum of frequency
                    val = freq / sm
                    Prob.append(val)  # probability

                    p = sum(Prob)  # sum of probability
                    for i in Prob:
                        gw = 1 + ((p * log2(val)) / (log2(len(data) + 1)))
                        GW.append(gw)
        for i in range(0, len(LW)):
            FW.append(LW[i] * GW[i])
        import pandas as pd
        dis = {'Weights': FW, 'words':words }
        df = pd.DataFrame(dis)

        params = {'purpose': 'Log Entropy Weights', 'data': df}
        return render(request, 'log_entropy.html', params)

    elif(rawf == "rawf"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
         آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
         ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
         اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
         بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
         تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
         جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
         جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
         دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
         رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
         سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
         فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
         لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
         مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
         نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
         وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
         چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
         کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
         کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
         گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
         ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
        """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)
        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text

        data = tokenization([x])
        unique_words = []
        for row in (data):
            for word in row:
                if len(word) >= 2 and word not in unique_words:
                    unique_words.append(word)
        unique_words.sort()
        vocab = {j: i for i, j in enumerate(unique_words)}

        from scipy.sparse import csr_matrix
        from collections import Counter
        from math import log2
        rows = []
        columns = []
        values = []
        words = []
        for idx, row in enumerate(data):  # for each document in the dataset
            # it will return a dict type object where key is the word and values is its frequency, {word:frequency}
            word_freq = dict(Counter(row))
            # for every unique word in the document
            for word, freq in word_freq.items():
                words.append(word)
                if len(word) < 2:
                    continue
                # we will check if its there in the vocabulary that we build in fit() function
                # dict.get() function will return the values, if the key doesn't exits it will return -1
                col_index = vocab.get(word, -1)  # retreving the dimension number of a word
                if col_index != -1:  # we are storing the index of the document
                    rows.append(idx)
                    # we are storing the dimensions of the word
                    columns.append(col_index)

                    # Compute Raw Frequency
                    val = freq
                    values.append(val)

        import pandas as pd
        dis = {'Weights': values, 'words': words}
        df = pd.DataFrame(dis)

        params = {'purpose': 'Raw Frequency Weights', 'data':df}
        return render(request, 'raw_freq.html', params)

    elif(rf == "rf"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
                 آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
                 ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
                 اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
                 بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
                 تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
                 جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
                 جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
                 دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
                 رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
                 سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
                 فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
                 لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
                 مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
                 نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
                 وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
                 چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
                 کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
                 کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
                 گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
                 ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
                """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text

        data = tokenization([x])
        unique_words = []
        for row in (data):
            for word in row:
                if len(word) >= 2 and word not in unique_words:
                    unique_words.append(word)
        unique_words.sort()
        vocab = {j: i for i, j in enumerate(unique_words)}

        from scipy.sparse import csr_matrix
        from collections import Counter
        from math import log2
        rows = []
        columns = []
        values = []
        sumf = []
        words = []
        for idx, row in enumerate(data):  # for each document in the dataset
            # it will return a dict type object where key is the word and values is its frequency, {word:frequency}
            word_freq = dict(Counter(row))
            # for every unique word in the document
            for word, freq in word_freq.items():
                words.append(word)
                if len(word) < 2:
                    continue
                # we will check if its there in the vocabulary that we build in fit() function
                # dict.get() function will return the values, if the key doesn't exits it will return -1
                col_index = vocab.get(word, -1)  # retreving the dimension number of a word
                if col_index != -1:  # we are storing the index of the document
                    rows.append(idx)
                    # we are storing the dimensions of the word
                    columns.append(col_index)

                    # Compute Relative Frequency
                    sfreq = freq
                    sumf.append(sfreq)
                    sm = sum(sumf)  # sum of frequency
                    val = freq / sm
                    values.append(val)  # probability

        import pandas as pd
        dis = {'Weights': values, 'words': words}
        df = pd.DataFrame(dis)

        params = {'purpose': 'Raw Frequency Weights','data':df}
        return render(request, 'relative_freq.html', params)

    elif(bf == "bf"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
                 آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
                 ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
                 اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
                 بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
                 تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
                 جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
                 جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
                 دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
                 رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
                 سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
                 فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
                 لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
                 مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
                 نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
                 وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
                 چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
                 کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
                 کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
                 گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
                 ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
                """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text

        data = tokenization([x])
        unique_words = []
        for row in (data):
            for word in row:
                if len(word) >= 2 and word not in unique_words:
                    unique_words.append(word)
        unique_words.sort()
        vocab = {j: i for i, j in enumerate(unique_words)}

        from scipy.sparse import csr_matrix
        from collections import Counter
        from math import log2
        rows = []
        columns = []
        values = []
        words = []
        for idx, row in enumerate(data):  # for each document in the dataset
            # it will return a dict type object where key is the word and values is its frequency, {word:frequency}
            word_freq = dict(Counter(row))
            # for every unique word in the document
            for word, freq in word_freq.items():
                words.append(word)
                if len(word) < 2:
                    continue
                # we will check if its there in the vocabulary that we build in fit() function
                # dict.get() function will return the values, if the key doesn't exits it will return -1
                col_index = vocab.get(word, -1)  # retreving the dimension number of a word
                if col_index != -1:  # we are storing the index of the document
                    rows.append(idx)
                    # we are storing the dimensions of the word
                    columns.append(col_index)

                    # Compute Binary Frequency
                    if freq > 0:
                        values.append(1)
                    elif freq == 0:
                        values.append(0)

        import pandas as pd
        dis = {'Weights': values, 'words': words}
        df = pd.DataFrame(dis)

        params = {'purpose': 'Binary Frequency Weights', 'data':df}
        return render(request, 'binary_freq.html', params)

    elif(acc == "acc"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
                         آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
                         ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
                         اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
                         بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
                         تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
                         جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
                         جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
                         دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
                         رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
                         سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
                         فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
                         لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
                         مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
                         نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
                         وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
                         چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
                         کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
                         کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
                         گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
                         ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
                        """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text


        import json

        # Opening JSON file
        with open('ACC.json', 'r') as openfile:
            # Reading from json file
            ACC_dict = json.load(openfile)

        list1 = []
        doc = tokenization([x])
        for i in doc:
            for word_ in i:
                weight = ACC_dict.get(word_, 0)
                list1.append(weight)
        words = []
        for lst in doc:
            for li in lst:
                words.append(li)

        import pandas as pd
        dic = {'Weights': list1, 'Words': words}
        df = pd.DataFrame(dic)
        params = {'purpose': 'ACC Weights', 'data': df}
        return render(request, 'acc.html', params)

    elif(acc2 == "acc2"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
            آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
            ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
            اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
            بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
            تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
            جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
            جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
            دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
            رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
            سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
            فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
            لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
            مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
            نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
            وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
            چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
            کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
            کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
            گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
            ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
                """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text

        import json

        # Opening JSON file
        with open('ACC2.json', 'r') as openfile:
            # Reading from json file
            ACC2_dict = json.load(openfile)


        list2 = []
        doc = tokenization([x])
        for i in doc:
            for word_ in i:
                weight = ACC2_dict.get(word_, 0)
                list2.append(weight)
        words = []
        for lst in doc:
            for li in lst:
                words.append(li)

        import pandas as pd
        dic = {'Weights': list2, 'Words': words}
        df = pd.DataFrame(dic)
        params = {'purpose': 'ACC2 Weights', 'data': df}
        return render(request, 'acc2.html', params)

    elif(ndm == "ndm"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
              آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
              ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
              اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
              بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
              تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
              جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
              جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
              دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
              رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
              سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
              فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
              لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
              مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
              نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
              وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
              چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
              کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
              کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
              گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
              ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
              """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text

        import json

        # Opening JSON file
        with open('NDM.json', 'r') as openfile:
            # Reading from json file
            NDM_dict = json.load(openfile)

        list3 = []
        doc = tokenization([x])
        for i in doc:
            for word_ in i:
                weight = NDM_dict.get(word_, 0)
                list3.append(weight)
        words = []
        for lst in doc:
            for li in lst:
                words.append(li)

        import pandas as pd
        dic = {'Weights': list3, 'Words': words}
        df = pd.DataFrame(dic)
        params = {'purpose': 'ACC2 Weights', 'data': df}
        return render(request, 'ndm.html', params)

    elif(odr == "odr"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
           آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
           ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
           اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
           بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
           تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
           جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
           جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
           دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
           رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
           سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
           فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
           لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
           مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
           نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
           وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
           چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
           کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
           کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
           گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
           ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
               """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text

        import json

        # Opening JSON file
        with open('OR.json', 'r') as openfile:
            # Reading from json file
            OR_dict = json.load(openfile)

        list4 = []
        doc = tokenization([x])
        for i in doc:
            for word_ in i:
                weight = OR_dict.get(word_, 0)
                list4.append(weight)
        words = []
        for lst in doc:
            for li in lst:
                words.append(li)

        import pandas as pd
        dic = {'Weights': list4, 'Words': words}
        df = pd.DataFrame(dic)
        params = {'purpose': 'Odds Ratio Weights', 'data': df}
        return render(request, 'odd.html', params)

    elif(gini == "gini"):
        from urduhack.normalization import normalize
        from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
        x = djtext
        x = normalize(x)
        x = remove_punctuation(x)
        x = remove_accents(x)
        x = replace_urls(x)
        x = replace_emails(x)
        x = replace_currency_symbols(x)
        x = normalize_whitespace(x)

        from typing import FrozenSet
        # Urdu Language Stop words list
        STOP_WORDS: FrozenSet[str] = frozenset("""
                   آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
                   ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
                   اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
                   بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
                   تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
                   جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
                   جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
                   دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
                   رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
                   سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
                   فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
                   لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
                   مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
                   نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
                   وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
                   چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
                   کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
                   کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
                   گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
                   ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
                       """.split())

        def remove_stopwords(text: str):
            return " ".join(word for word in text.split() if word not in STOP_WORDS)

        x = remove_stopwords(x)

        import stanfordnlp
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='ur')

        def tokenization(data):
            def token(data):
                doc = nlp(data)
                for i, sentence in enumerate(doc.sentences):
                    a = [f"{token.text}" for token in sentence.tokens]
                return a

            tokenized_text = []
            for document in data:
                docum = token(document)
                tokenized_text.append(docum)
            return tokenized_text

        import json

        # Opening JSON file
        with open('GINI.json', 'r') as openfile:
            # Reading from json file
            GINI_dict = json.load(openfile)

        list5 = []
        doc = tokenization([x])
        for i in doc:
            for word_ in i:
                weight = GINI_dict.get(word_, 0)
                list5.append(weight)
        words = []
        for lst in doc:
            for li in lst:
                words.append(li)

        import pandas as pd
        dic = {'Weights': list5, 'Words': words}
        df = pd.DataFrame(dic)
        params = {'purpose': 'Odds Ratio Weights', 'data': df}
        return render(request, 'gini.html', params)
    else:
        return HttpResponse('Error')


def classify(request):
    djtext = request.GET.get('text', 'default')
    from urduhack.normalization import normalize
    from urduhack.preprocessing import normalize_whitespace, \
            remove_punctuation, remove_accents, replace_urls, \
            replace_emails, replace_currency_symbols
    x = djtext
    x = normalize(x)
    x = remove_punctuation(x)
    x = remove_accents(x)
    x = replace_urls(x)
    x = replace_emails(x)
    x = replace_currency_symbols(x)
    x = normalize_whitespace(x)

    from tensorflow.keras.preprocessing.text import one_hot
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers

    voc_size = 18000
    model1 = Sequential()
    model1.add(layers.Embedding(voc_size, 300, input_length=500))
    model1.add(layers.Conv1D(128, 3, activation='relu'))
    model1.add(layers.GlobalMaxPooling1D())
    model1.add(layers.Dense(128, activation='relu'))
    model1.add(layers.Dense(1, activation='sigmoid'))
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model1.load_weights('OneHot_with_CNN.h5')
    djtext = x
    o = [one_hot(djtext, voc_size)]
    x = pad_sequences(o, padding='pre', maxlen=500)
    yhat = (model1.predict(x) > 0.5).astype('int32')
    if yhat == 1:
        status = "THIS NEWS IS REAL"
    else:
        status = "THIS NEWS IS FAKE"


    params = {'purpose': 'Status of The News', 'data': status}
    return render(request, 'Main_Home.html', params)