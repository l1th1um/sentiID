from __future__ import division
import sys
import re
import json
import pkg_resources

class DictSentiment:
    def __init__(self):
        self.file_wl = pkg_resources.resource_filename('sentiID','data/sentiwordnet_avg_dict_embedSG_oltw_tuned.txt')
        self.file_emo = pkg_resources.resource_filename('sentiID','data/sentiment_emotions.txt')
        self.emo = True

    def create_sentiwordlist(self, file, boundary=False):   
        S = {}
        with open(file) as f:
            for line in f:
                row = line.split("\t")
                word = row[0]
                score_pos = row[1].strip()
                score_neg = row[2].strip()
                S[word] = []
                S[word].append(score_pos)
                S[word].append(score_neg)
        
        token = list(S)
        token.sort(key=lambda word: len(word), reverse=True)
        token = [re.escape(word) for word in token]
        regex = '(?:' + "|".join(token) + ')'
        if boundary:
            regex = r"\b" + regex + r"\b"
        return regex, S
    
    def replaceTwoOrMore(self, s):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        pattern = pattern.sub(r"\1", s)
        return pattern

    def checkValence(self,inp_text):
        regex_wordlist, S = self.create_sentiwordlist(self.file_wl, boundary=True)
        if self.emo:
            regex_emoticons, E = self.create_sentiwordlist(self.file_emo)
            regex = '(' + regex_wordlist + '|' + regex_emoticons + ')'
            S.update(E)
        else:
            regex = '(' + regex_wordlist + ')'
        wordlist = re.compile(regex, flags=re.UNICODE)
        
        negasi = ["tidak", "enggak", "nggak", "engga", "ga", "gak", "gk", "gag", "bukan", "tiada", "non", "tak", "kagak", "kaga", "ngga","ndak", "kurang","jangan", "jgn", "janganlah"]
        b_booster = ["teramat","amat","sangat", "lebih"]
        e_booster = ["banget","bgt","sekali"]

        text = inp_text.lower().strip()
        text = self.replaceTwoOrMore(text)
        token = wordlist.findall(text) 
        #print token 
           
        sentiment_pos = 0
        sentiment_neg = 0
        tk = {}
        deb_word = {}
        for idx,w in enumerate(token):
            try:
                val_pos = float(S[w][0])
                val_neg = float(S[w][1])
                
                prev_sub = '\w+\W+' + re.escape(w)
                next_sub = re.escape(w) + '\W+\w+'
                prev_word = re.findall(prev_sub,text)[0].split()[0] if re.findall(prev_sub,text) else ""
                next_word = re.findall(next_sub,text)[0].split()[-1] if re.findall(next_sub,text) else ""

                if prev_word in negasi:
                    val_pos = float(S[w][1])
                    val_neg = float(S[w][0])
                elif (prev_word in b_booster) or (next_word in e_booster):
                    if val_pos>val_neg:
                        val_pos = 1
                        val_neg = 0
                    elif val_pos<val_neg:
                        val_pos = 0
                        val_neg = 1
                deb_word[w]={}
                deb_word[w]["pos"] = val_pos
                deb_word[w]["neg"] = val_neg
                sentiment_pos += float(val_pos)
                sentiment_neg += float(val_neg)
            except:
                #pass
                e = sys.exc_info()[1]
                print ("Error: ", e)
        return deb_word, sentiment_pos, sentiment_neg

    def debug(self,inp_text):
        deb_data,sentiment_pos, sentiment_neg = self.checkValence(inp_text)
        return json.dumps(deb_data)

    def predict(self, inp_text): 
        deb_data,sentiment_pos, sentiment_neg = self.checkValence(inp_text)
        if sentiment_pos>sentiment_neg:
            sentiment_pred = "positif"
        elif sentiment_pos<sentiment_neg:
            sentiment_pred = "negatif"
        else:
            sentiment_pred = "netral"
        txt_pred = sentiment_pred + " (pos: " + str(sentiment_pos) + ", neg: " + str(sentiment_neg) + ")"
        return txt_pred
