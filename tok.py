import re
import matplotlib.pyplot as plt
import math
import random
corpus2=[]
unigrams2={}
bigrams2={}
laplace_unigrams_prob2={}
trigrams2={}
def tokenise(file):
    corpus=[]
    input_file = open(file,"r")
    #input_file = open("movies.txt","r")
    #input_file = open("news.txt","r")
    file_content=input_file.readlines()
    
    for line in file_content:
        
        #regex = re.compile("http[s]?[\s]?:[\s]?//[a-zA-Z0-9/\.]+")
        line=re.sub('[^a-zA-Z0-9 ]', '', line)
        regex = re.compile('[a-zA-Z0-9]+')
        tokens=regex.findall(line)
        corpus2.append(tokens)
        corpus.append(tokens)
    printvalues(corpus)
    input_file.close()
    #print corpus
    return corpus
def printvalues(value):
    # print value
    pass
def sort_dict(d):
    return sorted(d.items(), key = lambda x : x[1], reverse = True)

def get_unigrams(corpus):
    printvalues(corpus)
    unigrams = {}
    for sentence in corpus:
        for word in sentence:
            printvalues(word)
            if word not in unigrams:
                unigrams[word] = 0
                unigrams2[word] = 0
                printvalues(word)
            unigrams[word] += 1
            unigrams2[word] += 1
        printvalues(sentence)

    unigrams_prob = {}
    N = sum(unigrams.values())
    for word in unigrams:
        unigrams_prob[word] = round( (unigrams[word]) / float(N), 15)
        printvalues(unigrams_prob[word])
    printvalues(sort_dict(unigrams))
    #plot(sort_dict(unigrams_prob))
    #plot_log_log1(sort_dict(unigrams_prob))
    return unigrams,unigrams_prob

def get_bigrams(corpus,unigrams):
    bigrams = {}
    for sentence in corpus:
        printvalues(sentence)
        for index, word in enumerate(sentence):
            if index > 0:
                pair  = (sentence[index - 1], word)
                printvalues(pair)
                if pair not in bigrams:
                    printvalues(pair)
                    bigrams[pair] =0
                    bigrams2[pair] =0
                bigrams[pair] += 1
                bigrams2[pair] +=1
                printvalues(pair)
    bigrams_prob={}
    for pair in bigrams:
        bigrams_prob[pair] = round( (bigrams[pair]) / float(unigrams[pair[0]]),  15)
    printvalues(sort_dict(bigrams))
    #plot(sort_dict(bigrams_prob))
    #plot_log_log(sort_dict(bigrams_prob))
    return bigrams,bigrams_prob

def get_trigrams(corpus, bigrams):
    trigrams = {}
    for sentence in corpus:
        printvalues(sentence)
        for index, word in enumerate(sentence):
            if index > 1:
                pair  = (sentence[index - 2],sentence[index - 1], word)
                printvalues(pair)
                if pair not in trigrams:
                    trigrams[pair] =0
                    trigrams2[pair]=0
                trigrams[pair] += 1
                trigrams2[pair] +=1
                printvalues(trigrams2[pair])
    trigrams_prob={}
    for pair in trigrams:
        bigrampair = float( bigrams[( pair[0],pair[1]) ] )
        trigrams_prob[pair] = round((trigrams[pair]) / bigrampair, 15)

        printvalues(trigrams_prob[pair])
    #print sort_dict(trigrams_prob)
    #plot(sort_dict(trigrams_prob))
    #plot_log_log(sort_dict(trigrams_prob))
    return trigrams,trigrams_prob

def get_laplace_unigrams(unigrams,V):
    laplace_unigrams_prob = {}

    N = sum(unigrams.values())
    printvalues(N)
    for word in unigrams:
        laplace_unigrams_prob[word] = (unigrams[word] + 1)/ float(N+V)
        laplace_unigrams_prob[word] = round(laplace_unigrams_prob[word], 15)
        printvalues(laplace_unigrams_prob[word])
    printvalues(sort_dict(laplace_unigrams_prob))
    #plot(sort_dict(laplace_unigrams_prob))
    #plot_log_log(sort_dict(laplace_unigrams_prob))
    return laplace_unigrams_prob


def get_laplace_bigrams(unigrams,bigrams,V):
    laplace_bigrams_prob = {}
    N = sum(unigrams.values())
    printvalues(N)
    for pair in bigrams:
        laplace_bigrams_prob[pair] = (bigrams[pair] + 1)/ float(unigrams[pair[0]]+V)
        laplace_bigrams_prob[pair] = round(laplace_bigrams_prob[pair], 15)
        printvalues(laplace_bigrams_prob[pair])
    printvalues(sort_dict(laplace_bigrams_prob))
    return laplace_bigrams_prob

def get_laplace_trigrams(unigrams,bigrams,trigrams,V):
    laplace_trigrams_prob = {}
    N = sum(unigrams.values())
    printvalues(N)
    for pair in trigrams:
        pairi=(pair[0],pair[1])
        laplace_trigrams_prob[pair] = (trigrams[pair] + 1)/ float(bigrams[pairi]+V)
        laplace_trigrams_prob[pair] = round(laplace_trigrams_prob[pair], 15)
        printvalues(laplace_trigrams_prob[pair])
    printvalues(sort_dict(laplace_trigrams_prob))
    return laplace_trigrams_prob

def get_wittenbell_unigrams(unigrams,unigrams_prob,V):
    wittenbell_unigrams_prob = {}
    N = sum(unigrams.values())
    T = len(unigrams.keys())
    printvalues(T)
    for word in unigrams:
        var= (N/float(N+T))
        wittenbell_unigrams_prob[word] = (var * unigrams_prob[word])
        wittenbell_unigrams_prob[word] = wittenbell_unigrams_prob[word] + (1-var)/V
        wittenbell_unigrams_prob[word] = round(wittenbell_unigrams_prob[word], 15)
        printvalues(wittenbell_unigrams_prob[word])
    printvalues(sort_dict(wittenbell_unigrams_prob))
    return wittenbell_unigrams_prob

def get_wittenbell_bigrams_variables(bigrams,word):
    printvalues(bigrams)
    distinct_1_in_1_bigrams=0
    total_1_in_1_bigrams=0
    printvalues(bigrams)
    for pair in bigrams:
        if word==pair[0]:
            printvalues(pair)
            distinct_1_in_1_bigrams+=1
            total_1_in_1_bigrams+=bigrams[pair]
            printvalues(bigrams[pair])
    return distinct_1_in_1_bigrams,total_1_in_1_bigrams

def get_wittenbell_bigrams(unigrams,bigrams,unigrams_prob,wittenbell_unigrams_prob):
    printvalues(unigrams)
    wittenbell_bigrams_prob = {}
    for pair in bigrams:
        printvalues(pair)
        distinct_1_in_1_bigrams,total_1_in_1_bigrams = get_wittenbell_bigrams_variables(bigrams,pair[0])
        x=distinct_1_in_1_bigrams/float(distinct_1_in_1_bigrams+total_1_in_1_bigrams)
        x=round(x, 15)
        wittenbell_bigrams_prob[pair] = (1-x) * bigrams_prob[pair]
        wittenbell_bigrams_prob[pair] += x * wittenbell_unigrams_prob[pair[0]]
    printvalues(sort_dict(wittenbell_bigrams_prob))
    return wittenbell_bigrams_prob

def get_wittenbell_trigrams_variables(trigrams,word):
    distinct_1_2_trigrams=0
    total_1_2_trigrams=0
    printvalues(trigrams)
    for pair in trigrams:
        if word[0]==pair[0] and word[1]==pair[1]:
            printvalues(pair)
            distinct_1_2_trigrams+=1
            total_1_2_trigrams+=trigrams[pair]
            printvalues(trigrams[pair])
    return distinct_1_2_trigrams,total_1_2_trigrams


def get_wittenbell_trigrams(unigrams,trigrams,trigrams_prob,wittenbell_bigrams_prob):
    printvalues(unigrams)
    wittenbell_trigrams_prob = {}
    for pair in trigrams:
        printvalues(pair)
        pairi=(pair[0],pair[1])
        distinct_1_2_trigrams,total_1_2_trigrams = get_wittenbell_trigrams_variables(trigrams,pair)
        printvalues(distinct_1_2_trigrams)
        x=distinct_1_2_trigrams/float(distinct_1_2_trigrams+total_1_2_trigrams)
        x=round(x, 15)
        wittenbell_trigrams_prob[pair] = (1-x) * trigrams_prob[pair] 
        wittenbell_trigrams_prob[pair] += x * wittenbell_bigrams_prob[pairi]
    printvalues(sort_dict(wittenbell_trigrams_prob))
    return wittenbell_trigrams_prob

def get_kn_unigrams(unigrams,V):
    kn_unigrams_prob={}
    d=0.75
    N=sum(unigrams.values())
    printvalues(N)
    for word in unigrams:
        kn_unigrams_prob[word] = (max(unigrams[word]-d, 0)/float(N)) 
        kn_unigrams_prob[word] += (d/float(V))
    printvalues(sort_dict(kn_unigrams_prob))
    return kn_unigrams_prob

def get_kn_bigrams_variables(bigrams,pair):
    bigrams_with_first_term=0
    bigrams_with_last_term=0
    for words in bigrams:
        printvalues(words)
        if words[0]==pair[0]:
            bigrams_with_first_term+=1
            printvalues(bigrams_with_first_term)
        if words[1]==pair[1]:
            bigrams_with_last_term+=1
    printvalues(bigrams_with_first_term)
    printvalues(bigrams_with_last_term)
    return bigrams_with_first_term,bigrams_with_last_term

def get_kn_bigrams(unigrams,bigrams):
    kn_bigrams_prob={}
    d=0.75
    printvalues(unigrams)
    total_bigrams=len(bigrams.keys())
    for pair in bigrams:
        printvalues(pair)
        bigrams_with_first_term,bigrams_with_last_term=get_kn_bigrams_variables(bigrams,pair)
        value1 = max(bigrams[pair]-d,0) / float(unigrams[pair[0]])
        value2 = bigrams_with_first_term/float(unigrams[pair[0]])
        value3 = bigrams_with_last_term/float(total_bigrams)
        kn_bigrams_prob[pair]= ( value1 ) + d * ( value2) * ( value3 )
        kn_bigrams_prob[pair] = round(kn_bigrams_prob[pair], 15)

    printvalues(sort_dict(kn_bigrams_prob))
    return kn_bigrams_prob

def get_kn_trigrams_variables(bigrams,trigrams,pair):
    trigrams_1_2_term=0
    trigrams_2_3_term=0
    trigrams_2_term=0
    printvalues(trigrams_2_term)
    bigrams_2_in_1_term=0
    bigrams_3_in_2_term=0
    printvalues(bigrams_3_in_2_term)
    for words in trigrams:
        if words[0]==pair[0] and words[1]==pair[1]:
            trigrams_1_2_term+=1
        if words[1]==pair[1] and words[2]==pair[2]:
            trigrams_2_3_term+=1
        if words[1]==pair[1]:
            trigrams_2_term+=1
        printvalues(words)
    for words in bigrams:
        if words[0]==pair[1]:
            bigrams_2_in_1_term+=1
        if words[1]==pair[2]:
            bigrams_3_in_2_term+=1
        printvalues(words)
    return trigrams_1_2_term,trigrams_2_3_term,trigrams_2_term,bigrams_2_in_1_term,bigrams_3_in_2_term


def get_kn_trigrams(unigrams,bigrams,trigrams):
    printvalues(unigrams)
    kn_trigrams_prob={}
    d=0.75
    total_bigrams=len(bigrams.keys())
    printvalues(total_bigrams)
    for pair in trigrams:
        pairi=(pair[0],pair[1])
        printvalues(pair)
        trigrams_1_2_term,trigrams_2_3_term,trigrams_2_term,bigrams_2_in_1_term,bigrams_3_in_2_term=get_kn_trigrams_variables(bigrams,trigrams,pair)
        value1 = max(trigrams[pair]-d,0) / float(bigrams[pairi])
        value2 = trigrams_1_2_term/float(bigrams[pairi])
        value3 = ( max(trigrams_2_3_term-d,0)/float(trigrams_2_term) )
        value4 = ( max(trigrams_2_3_term-d,0)/float(trigrams_2_term) )
        kn_trigrams_prob[pair]= ( value1 ) + d*(value2 )*( value3 + d * value4*(bigrams_3_in_2_term/float(total_bigrams)) )
        kn_trigrams_prob[pair] = round(kn_trigrams_prob[pair], 15)
    printvalues(sort_dict(kn_trigrams_prob))
    return kn_trigrams_prob


def plot(dict1,dict2,dict3,dict4,l1,l2,l3,l4,name):
    x1 = []
    y1 = []
    printvalues(len(dict1))
    for i in range(len(dict1)):
        y1.append((dict1[i][1]))
        printvalues(y1)
        x1.append((i))
        printvalues(x1)
        
    plt.plot(x1,y1,label=l1)

    x2 = []
    y2 = []
    printvalues(len(dict2))
    for i in range(len(dict2)):
        y2.append((dict2[i][1]))
        printvalues(y2)
        x2.append((i))
        printvalues(x2)
    plt.plot(x2,y2,label=l2)

    x3 = []
    y3 = []
    printvalues(len(dict3))
    for i in range(len(dict3)):
        y3.append((dict3[i][1]))
        printvalues(y3)
        x3.append((i))
        printvalues(x3)
    plt.plot(x3,y3,label=l3)

    x4 = []
    y4 = []
    printvalues(len(dict4))
    for i in range(len(dict4)):
        y4.append((dict4[i][1]))
        printvalues(y4)
        x4.append((i))
        printvalues(x4)
    plt.plot(x4,y4,label=l4)

    plt.title(name)

    plt.legend()
    plt.show()

def plot_log_log1(dicti):
    new_ticks = []
    x = []
    y = []
    printvalues(len(dicti))
    for i in range(1,len(dicti)):
        new_ticks.append(dicti[i][0])
        y.append(math.log(dicti[i][1]))
        x.append(math.log(i))
        
    printvalues(x,y)
    plt.plot(x,y)
    #plt.xticks(x, new_ticks)
    plt.show()



def plot_log_log(dict1,dict2,dict3,dict4,l1,l2,l3,l4,name):
    printvalues(len(dict1))
    x1 = []
    y1 = []

    for i in range(len(dict1)):
    	if  i>=1:
            y1.append(math.log(dict1[i][1]))
            printvalues(y1)
            x1.append(math.log(i))
            printvalues(x1)
    plt.plot(x1,y1,label=l1)

    x2 = []
    y2 = []
    printvalues(len(dict2))
    for i in range(len(dict2)):
    	if  i>=1:
            y2.append(math.log(dict2[i][1]))
            printvalues(y2)
            x2.append(math.log(i))
            printvalues(x2)
    plt.plot(x2,y2,label=l2)

    x3 = []
    y3 = []
    printvalues(len(dict3))
    for i in range(len(dict3)):
    	if  i>=1:
            y3.append(math.log(dict3[i][1]))
            printvalues(y3)
            x3.append(math.log(i))
            printvalues(x3)
    plt.plot(x3,y3,label=l3)

    x4 = []
    y4 = []
    printvalues(len(dict4))
    for i in range(len(dict4)):
    	if  i>=1:
            y4.append(math.log(dict4[i][1]))
            printvalues(y4)
            x4.append(math.log(i))
            printvalues(x4)
    plt.plot(x4,y4,label=l4)
    plt.title(name)
    plt.legend()
    plt.show()
def estimated_count_wb(unigrams, bigrams, trigrams, prob_wb_unigrams, prob_wb_bigrams, prob_wb_trigrams):
    #For unigrams
    printvalues(unigrams)
    count_wb_unigrams={}
    N = sum(unigrams.values())
    for i in unigrams:
        count_wb_unigrams[i] = prob_wb_unigrams[i]
        count_wb_unigrams[i] = count_wb_unigrams[i] * N

    printvalues(count_wb_unigrams)
    #For bigrams
    count_wb_bigrams={}
    for (i,j) in bigrams:
        count_wb_bigrams[(i,j)] = prob_wb_bigrams[(i,j)] 
        count_wb_bigrams[(i,j)] = count_wb_bigrams[(i,j)]* unigrams[i]
    printvalues(count_wb_bigrams)

    #For trigrams
    count_wb_trigrams={}
    for (i, j, k) in trigrams:
        count_wb_trigrams[(i,j,k)] = prob_wb_trigrams[(i,j,k)] 
        count_wb_trigrams[(i,j,k)] = count_wb_trigrams[(i,j,k)]* bigrams[(i,j)]
    printvalues(count_wb_trigrams)

    return count_wb_unigrams, count_wb_bigrams, count_wb_trigrams


def estimated_count_laplace(unigrams, bigrams, trigrams, prob_smoothed_unigram_2000, prob_smoothed_bigram_2000, prob_smoothed_trigram_2000):

    #For unigrams
    count_laplace_unigrams = {}
    printvalues(unigrams)
    N = sum(unigrams.values())
    for i in unigrams:
        count_laplace_unigrams[i] = prob_smoothed_unigram_2000[i]
        count_laplace_unigrams[i] = count_laplace_unigrams[i] * N
    printvalues(count_laplace_unigrams)
    #For bigrams
    count_laplace_bigrams={}
    for (i,j) in bigrams:
        count_laplace_bigrams[(i,j)] = prob_smoothed_bigram_2000[(i,j)] 
        count_laplace_bigrams[(i,j)] = count_laplace_bigrams[(i,j)]* unigrams[i]
    printvalues(count_laplace_bigrams)
    #For trigrams
    count_laplace_trigrams={}
    for (i,j,k) in trigrams:
        count_laplace_trigrams[(i,j,k)] = prob_smoothed_trigram_2000[(i,j,k)] 
        count_laplace_trigrams[(i,j,k)] = count_laplace_trigrams[(i,j,k)] * bigrams[(i,j)]
    printvalues(count_laplace_trigrams)

    return count_laplace_unigrams, count_laplace_bigrams, count_laplace_trigrams
def estimated_count_kn(unigrams, bigrams, trigrams, prob_smoothed_unigram_2000, prob_smoothed_bigram_2000, prob_smoothed_trigram_2000):

    #For unigrams
    count_laplace_unigrams = {}
    N = sum(unigrams.values())
    for i in unigrams:
        count_laplace_unigrams[i] = prob_smoothed_unigram_2000[i]
        count_laplace_unigrams[i] = count_laplace_unigrams[i] * N
    printvalues(count_laplace_unigrams)
    #For bigrams
    count_laplace_bigrams={}
    for (i,j) in bigrams:
        count_laplace_bigrams[(i,j)] = prob_smoothed_bigram_2000[(i,j)] 
        count_laplace_bigrams[(i,j)] = count_laplace_bigrams[(i,j)]* unigrams[i]
    printvalues(count_laplace_bigrams)
    #For trigrams
    count_laplace_trigrams={}
    for (i,j,k) in trigrams:
        count_laplace_trigrams[(i,j,k)] = prob_smoothed_trigram_2000[(i,j,k)] 
        count_laplace_trigrams[(i,j,k)] = count_laplace_trigrams[(i,j,k)] * bigrams[(i,j)]
    printvalues(count_laplace_trigrams)

    return count_laplace_unigrams, count_laplace_bigrams, count_laplace_trigrams

def get_kn_trigrams_laplace(unigrams,bigrams,trigrams,unigrams_lap,bigrams_lap,trigrams_lap):
    kn_trigrams_prob={}
    d=0.75
    total_bigrams=len(bigrams.keys())
    for pair in trigrams:
        pairi=(pair[0],pair[1])
        trigrams_1_2_term,trigrams_2_3_term,trigrams_2_term,bigrams_2_in_1_term,bigrams_3_in_2_term=get_kn_trigrams_variables(bigrams,trigrams,pair)
        value1 = ( max(trigrams_lap[pair]-d,0) / float(bigrams_lap[pairi]) )
        value2 = trigrams_1_2_term/float(bigrams[pairi])
        value3 =( max(trigrams_2_3_term-d,0)/float(trigrams_2_term) )
        value4 =bigrams_2_in_1_term/float(trigrams_2_term)
        value5 = bigrams_3_in_2_term/float(total_bigrams)
        kn_trigrams_prob[pair]= value1 + d*(value2 )*( value3 + d * (value4)*(value5) )
        kn_trigrams_prob[pair] = round(kn_trigrams_prob[pair], 15)
        printvalues(kn_trigrams_prob[pair])
    printvalues(sort_dict(kn_trigrams_prob))
    return kn_trigrams_prob

def get_kn_bigrams_laplace(unigrams,bigrams,unigrams_lap,bigrams_lap):
    kn_bigrams_prob={}
    d=0.75
    total_bigrams=len(bigrams.keys())
    for pair in bigrams:
        bigrams_with_first_term,bigrams_with_last_term=get_kn_bigrams_variables(bigrams,pair)
        value1 = max(bigrams_lap[pair]-d,0) / float(unigrams_lap[pair[0]])
        value2 = (bigrams_with_first_term/float(unigrams[pair[0]]) )
        value3 = ( bigrams_with_last_term/float(total_bigrams))
        kn_bigrams_prob[pair]= ( value1 ) + d * value2 * value3
        kn_bigrams_prob[pair] = round(kn_bigrams_prob[pair], 15)

    printvalues(sort_dict(kn_bigrams_prob))
    return kn_bigrams_prob

def get_kn_unigrams_laplace(unigrams,V,unigrams_lap):
    kn_unigrams_prob={}
    d=0.75
    N=sum(unigrams.values())
    printvalues(N)
    for word in unigrams:
        kn_unigrams_prob[word] = (max(unigrams_lap[word]-d, 0)/float(N)) 
        kn_unigrams_prob[word] += (d/float(V))
    printvalues(sort_dict(kn_unigrams_prob))
    return kn_unigrams_prob

def cond_bigrams(bigrams, key):
    
    joint = {k[1] : v for k, v in bigrams.items() if k[0] == key}
    sum_count = sum(joint.values())
    return {k : v / float(sum_count) for k, v in joint.items() }

def generate_bigrams(unigrams, bigrams, length=5, first_word = None):
    words = []
    if first_word == None:
        first_word = list(unigrams.keys())[random.randrange(0, len(unigrams))]
    words.append(first_word)
    for i in range(length - 1):
        prev = words[i]
        prev_dict = cond_bigrams(bigrams, prev)
        
        next_word = sorted(prev_dict.items(), key = lambda x : x[1], reverse = True)[0]
        words.append(next_word[0])
    return words
def cond_trigrams(trigrams, key):
    
    joint = {k[2] : v for k, v in trigrams.items() if (k[0] == key[0] and k[1] == key[1])}
    sum_count = sum(joint.values())
    return {k : v / float(sum_count) for k, v in joint.items() }

def generate_trigrams(unigrams, bigrams,trigrams, length=5, first_word = None):
    words = []
    if first_word == None:
        first_word = list(bigrams.keys())[random.randrange(0, len(bigrams))]
    words=(list(first_word))
    print words
    for i in range(length - 2):
        prev = words[i+1]
        prev2 = words[i]
        prev_dict = cond_trigrams(trigrams, [prev2,prev])
        
        next_word = sorted(prev_dict.items(), key = lambda x : x[1], reverse = True)[0]
        words.append(next_word[0])
    return words


#corpus=tokenise()

#unigrams,unigrams_prob=get_unigrams(corpus)

#laplace_unigrams_prob = get_laplace_unigrams(unigrams,200)
#printvalues(laplace_unigrams_prob)
#laplace_unigrams_prob2 = get_laplace_unigrams(unigrams,2000)
#printvalues(laplace_unigrams_prob2)
#laplace_unigrams_prob3 = get_laplace_unigrams(unigrams,len(unigrams))
#printvalues(laplace_unigrams_prob3)
#wittenbell_unigrams_prob = get_wittenbell_unigrams(unigrams,unigrams_prob,200)
#plot(sort_dict(wittenbell_unigrams_prob1),sort_dict(wittenbell_unigrams_prob2),sort_dict(wittenbell_unigrams_prob3),sort_dict(wittenbell_unigrams_prob4),"200","2000",len(unigrams),10*len(unigrams),"zipf_unigrams_diff_voc_witten_anime")
#plot_log_log(sort_dict(laplace_unigrams_prob1),sort_dict(laplace_unigrams_prob2),sort_dict(laplace_unigrams_prob3),sort_dict(laplace_unigrams_prob4),"200","2000",len(unigrams),10*len(unigrams),"log_unigrams_diff_voc_news")
#kn_unigrams_prob = get_kn_unigrams(unigrams,200)


#bigrams,bigrams_prob = get_bigrams(corpus,unigrams)
#printvalues(bigrams)
#printvalues(bigrams_prob)
#laplace_bigrams_prob1 = get_laplace_bigrams(unigrams,bigrams,200)
#laplace_bigrams_prob2 = get_laplace_bigrams(unigrams,bigrams,2000)
#printvalues(laplace_bigrams_prob2)
#laplace_bigrams_prob3 = get_laplace_bigrams(unigrams,bigrams,len(unigrams))
#printvalues(laplace_bigrams_prob3)
#laplace_bigrams_prob4 = get_laplace_bigrams(unigrams,bigrams,10*len(unigrams))
#wittenbell_bigrams_prob = get_wittenbell_bigrams(unigrams,bigrams,unigrams_prob,wittenbell_unigrams_prob)
#plot(sort_dict(laplace_bigrams_prob1),sort_dict(laplace_bigrams_prob2),sort_dict(laplace_bigrams_prob3),sort_dict(laplace_bigrams_prob4),"200","2000",len(unigrams),10*len(unigrams),"zipf_bigrams_diff_voc_anime")
#plot_log_log(sort_dict(laplace_bigrams_prob1),sort_dict(laplace_bigrams_prob2),sort_dict(laplace_bigrams_prob3),sort_dict(laplace_bigrams_prob4),"200","2000",len(unigrams),10*len(unigrams),"log_bigrams_diff_voc_anime")
#kn_bigrams_prob =  get_kn_bigrams(unigrams,bigrams)
#printvalues(kn_bigrams_prob)


#trigrams,trigrams_prob = get_trigrams(corpus,bigrams)
#printvalues(trigrams)
#laplace_trigrams_prob1 = get_laplace_trigrams(unigrams,bigrams,trigrams,200)
#laplace_trigrams_prob2 = get_laplace_trigrams(unigrams,bigrams,trigrams,2000)
#printvalues(laplace_trigrams_prob2)
#laplace_trigrams_prob3 = get_laplace_trigrams(unigrams,bigrams,trigrams,len(unigrams))
#laplace_trigrams_prob4 = get_laplace_trigrams(unigrams,bigrams,trigrams,10*len(unigrams))
#wittenbell_trigrams_prob = get_wittenbell_trigrams(unigrams,trigrams,trigrams_prob,wittenbell_bigrams_prob)
#plot(sort_dict(laplace_bigrams_prob3),sort_dict(wittenbell_bigrams_prob),sort_dict(kn_bigrams_prob),sort_dict(laplace_bigrams_prob4),"laplace","wittenbell","kneney","laplacefinal","zipf_trigrams_diff_voc_anime")
#plot_log_log(sort_dict(laplace_trigrams_prob1),sort_dict(laplace_trigrams_prob2),sort_dict(laplace_trigrams_prob3),sort_dict(laplace_trigrams_prob4),"200","2000",len(unigrams),10*len(unigrams),"log_trigrams_diff_voc_anime")
#kn_trigrams_prob =  get_kn_trigrams(unigrams,bigrams,trigrams)
#printvalues(kn_trigrams_prob)

#kn_unigrams,kn_bigrams,kn_trigrams=estimated_count_kn(unigrams, bigrams, trigrams, kn_unigrams_prob, kn_bigrams_prob, kn_trigrams_prob)
#laplace_unigrams,laplace_bigrams,laplace_trigrams = estimated_count_laplace(unigrams, bigrams, trigrams, laplace_unigrams_prob2, laplace_bigrams_prob2, laplace_trigrams_prob2)
#key_trigrams_laplace = get_kn_trigrams_laplace(unigrams,bigrams,trigrams,laplace_unigrams,laplace_bigrams,laplace_trigrams)

#key_bigrams_laplace = get_kn_bigrams_laplace(unigrams,bigrams,laplace_unigrams,laplace_bigrams)
#plot(sort_dict(kn_trigrams_prob),sort_dict(key_trigrams_laplace),sort_dict({}),sort_dict({}),"key_trigram","laplace+key_trigram","nothing","nothing","key+laplace_trigrams")
#plot(sort_dict(kn_bigrams_prob),sort_dict(key_bigrams_laplace),sort_dict({}),sort_dict({}),"key_bigram","laplace+key_bigram","nothing","nothing","key+laplace_bigrams")
#print generate_bigrams(kn_unigrams, kn_bigrams, length=5, first_word = None)
#print generate_trigrams(kn_unigrams, kn_bigrams,kn_trigrams, length=5, first_word = None)
corpus_anime = tokenise("anime.txt")
corpus_movies = tokenise("movies.txt")
corpus_news = tokenise("news.txt")

unigrams_anime,unigrams_prob_anime = get_unigrams(corpus_anime)
bigrams_anime,bigrams_prob_anime = get_bigrams(corpus_anime,unigrams_anime)
trigrams_anime,trigrams_prob_anime = get_trigrams(corpus_anime,bigrams_anime)

unigrams_movies,unigrams_prob_movies = get_unigrams(corpus_movies)
bigrams_movies,bigrams_prob_movies = get_bigrams(corpus_movies,unigrams_movies)
trigrams_movies,trigrams_prob_movies = get_trigrams(corpus_movies,bigrams_movies)

unigrams_news,unigrams_prob_news = get_unigrams(corpus_news)
bigrams_news,bigrams_prob_news = get_bigrams(corpus_news,unigrams_news)
trigrams_news,trigrams_prob_news = get_trigrams(corpus_news,bigrams_news)

plot(sort_dict(unigrams_anime),sort_dict(unigrams_movies),sort_dict(unigrams_news),{},"anime_uni","movies_uni","news_uni","none","Unigrams Zipfs")
plot(sort_dict(bigrams_anime),sort_dict(bigrams_movies),sort_dict(bigrams_news),{},"anime_bi","movies_bi","news_bi","none","Bigrams Zipfs")
plot(sort_dict(trigrams_anime),sort_dict(trigrams_movies),sort_dict(trigrams_news),{},"anime_tri","movies_tri","news_tri","none","Trigrams Zipfs")

