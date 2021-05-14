import nltk
from nltk.probability import FreqDist
from nltk.corpus import brown
from collections import Counter
import operator

def remove_stop_words(phraseList):
    stopwords = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above",
                 "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added",
                 "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against",
                 "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already",
                 "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce",
                 "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways",
                 "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately",
                 "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking",
                 "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b",
                 "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming",
                 "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being",
                 "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol",
                 "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but",
                 "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause",
                 "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj",
                 "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning",
                 "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding",
                 "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu",
                 "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely",
                 "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj",
                 "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards",
                 "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed",
                 "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else",
                 "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es",
                 "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody",
                 "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa",
                 "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five",
                 "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly",
                 "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further",
                 "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives",
                 "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy",
                 "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't",
                 "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her",
                 "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes",
                 "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how",
                 "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6",
                 "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il",
                 "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc",
                 "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead",
                 "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd",
                 "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt",
                 "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko",
                 "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least",
                 "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj",
                 "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma",
                 "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime",
                 "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml",
                 "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug",
                 "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne",
                 "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never",
                 "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non",
                 "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now",
                 "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od",
                 "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once",
                 "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others",
                 "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow",
                 "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part",
                 "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi",
                 "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly",
                 "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily",
                 "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que",
                 "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really",
                 "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related",
                 "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf",
                 "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry",
                 "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second",
                 "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
                 "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she",
                 "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show",
                 "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar",
                 "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some",
                 "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat",
                 "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr",
                 "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such",
                 "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken",
                 "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks",
                 "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them",
                 "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein",
                 "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these",
                 "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think",
                 "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three",
                 "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to",
                 "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly",
                 "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue",
                 "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until",
                 "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness",
                 "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via",
                 "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants",
                 "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b",
                 "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats",
                 "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby",
                 "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim",
                 "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose",
                 "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won",
                 "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x",
                 "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes",
                 "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself",
                 "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]

    # s = set(stopwords.words('english'))
    output = []
    for w in phraseList:
        if w not in stopwords:
            output.append(w)

    return output


def process_by_frequency(phraseDict):
    """
    A function that removes the words in phrase dictionary that has a frequency lower than its
    frequency in the Brown corpus. Returns a phrase list after the removals.
    """
    phraseList = []

    # get the total number of words from phrase list dictionary
    total_phraseList_count = 0
    for word, count in phraseDict.items():
        total_phraseList_count += count

    # a word count dictionary for all brown corpus words
    corpus_word_dict = {}
    brown_total = 0
    for sentence in brown.sents():
        for word in sentence:
            brown_total += 1
            count = corpus_word_dict.get(word.lower(), -1)
            if count == -1:
                corpus_word_dict[word.lower()] = 1
            else:
                corpus_word_dict[word.lower()] = count + 1

    # remove words in phrase list that has less frequency that in brown corpus
    for word, count in phraseDict.items():
        phraseDict_freq = count / total_phraseList_count
        brown_count = corpus_word_dict.get(word, -1)
        if brown_count == -1:
            phraseList.append(word)
        else:
            brown_freq = brown_count / brown_total
            if phraseDict_freq > brown_freq:
                phraseList.append(word)
            else:
                pass

    return phraseList


def sequential_pattern_mining(transcations, min_support):
    """
    A function that extracts the frequent itemsets from the phrase lists from OCR
    """
    def make_n(last, second, n):
        list_last = list(last)
        output = []
        for name in last:
            for second_name in second:
                if name[n - 1] == second_name[0]:
                    temp_list = list(name)
                    temp_list.append(second_name[1])
                    output.append(tuple(temp_list))
        return output

    def n_check(trans, names, n):
        n_list = {}
        for name in names:
            counter = 0
            for sentence in trans:
                for i in range(len(sentence) - n + 1):
                    has = True
                    for j in range(n):
                        if sentence[i + j] != name[j]:
                            has = False
                    if (has):
                        counter += 1
            if (counter >= 2):
                n_list[name] = counter
        return n_list

    double_base_counter = Counter()
    for sentence in transcations:
        for i in range(len(sentence) - 1):
            temp = (sentence[i], sentence[i + 1])
            double_base_counter[temp] += 1

    base_list2 = dict((word_pair, double_base_counter[word_pair]) for word_pair in double_base_counter if
                      double_base_counter[word_pair] >= min_support)
    second_support_names = list(base_list2)

    last = base_list2
    count = 2
    output = base_list2.copy()
    while (len(last) > 0 and count < 5):
        names_new = make_n(last, second_support_names, count)
        last = n_check(transcations, names_new, count + 1)
        output.update(last)
        count += 1

    sorted_a = sorted(output.items())
    sorted_value = sorted(sorted_a, key=operator.itemgetter(1), reverse=True)

    return sorted_value


def sequential_pattern_mining_1(transcations, min_support):
    """
    A function that extracts the frequent itemsets from the phrase lists from OCR
    """

    def make_n(last, second, n, makedict):
        list_last = list(last)
        output = []
        for name in last:
            for second_name in second:
                if name[n - 1] == second_name[0]:
                    temp_list = list(name)
                    tempcpy = temp_list.copy()
                    temp_list.append(second_name[1])
                    output.append(tuple(temp_list))
                    t = frozenset(temp_list)
                    makedict[t] = tempcpy
        return output

    def n_check(trans, names, n):
        n_list = {}
        removelist = []
        subset = []
        for name in names:
            counter = 0
            for sentence in trans:
                for i in range(len(sentence) - n + 1):
                    has = True
                    for j in range(n):
                        if sentence[i + j] != name[j]:
                            has = False
                    if (has):
                        counter += 1
            if (counter >= 2):
                fname = frozenset(name)
                son = makedict[fname]
                son = [son]
                for i in range(len(name) - 1):
                    if i == 0:
                        continue
                    subset = subset + [name[i:len(name)]]
                removelist = removelist + son
                n_list[name] = counter
        return n_list, removelist, subset

    double_base_counter = Counter()
    makedict = {}
    for sentence in transcations:
        for i in range(len(sentence) - 1):
            temp = (sentence[i], sentence[i + 1])
            double_base_counter[temp] += 1

    base_list2 = dict((word_pair, double_base_counter[word_pair]) for word_pair in double_base_counter if
                      double_base_counter[word_pair] >= min_support)
    second_support_names = list(base_list2)

    last = base_list2
    count = 2
    output = base_list2.copy()
    while (len(last) > 0 and count < 5):
        names_new = make_n(last, second_support_names, count, makedict)
        last, removelist, subs = n_check(transcations, names_new, count + 1)
        output.update(last)
        for i in removelist:
            del output[tuple(i)]
        for i in subs:
            if i in output:
                del output[i]
        count += 1

    sorted_a = sorted(output.items())
    sorted_value = sorted(sorted_a, key=operator.itemgetter(1), reverse=True)

    return sorted_value


def patterns_to_list(frequent_patterns):
    """
    A function that takes a phrase dictionary and converts it to the input transactions for sequential pattern mining
    """
    frequent_pattern_list = []
    for words, count in frequent_patterns:
        phrase = ' '.join(words)
        frequent_pattern_list.append(phrase)

    return frequent_pattern_list
