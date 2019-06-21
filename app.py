from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
from flaskext.mysql import MySQL
import NLPprocess

mysql = MySQL()
app = Flask(__name__, static_folder='static')
# mysql configuratoin
app.config['MYSQL_DATABASE_HOST']       = 'localhost'
app.config['MYSQL_DATABASE_USER']       = 'root'
app.config['MYSQL_DATABASE_PASSWORD']   = 'password'
app.config['MYSQL_DATABASE_DB']         = 'pilpres'
mysql.init_app(app)

# app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/jokowi')
def jokowi():
	data_testing = NLPprocess.get_data_from_twitter("jokowi")
	caseholding = NLPprocess.function_case_holding(data_testing)
	cleansing = NLPprocess.function_cleansing_tweet(caseholding, data_testing)
	normalization = NLPprocess.function_normalization(cleansing)
	negation = NLPprocess.function_negation_tweet(normalization)
	tokenize = NLPprocess.function_tokenize_tweet(negation)
	stopword = NLPprocess.funtion_stopword_tweet(tokenize)
	pos_tagging = NLPprocess.function_pos_tagging(stopword)
	stemmer = NLPprocess.function_stemmer_tweet(pos_tagging)
	lexicon_based = NLPprocess.funtion_lexicon_based(stemmer)
	naive_bayes = NLPprocess.function_naive_bayes(lexicon_based)
	information_classifier = NLPprocess.function_information_classifier(naive_bayes)

	return render_template('jokowi.html', crawling = data_testing, caseHolding = caseholding, cleansing = cleansing, normalization = normalization, negation = negation, tokenize = tokenize, stopwords = stopword, pos = pos_tagging, stemming = stemmer, lexicon = lexicon_based, naive = naive_bayes, information = information_classifier)

@app.route('/prabowo')
def prabowo():
	data_testing = NLPprocess.get_data_from_twitter("prabowo")
	caseholding = NLPprocess.function_case_holding(data_testing)
	cleansing = NLPprocess.function_cleansing_tweet(caseholding, data_testing)
	normalization = NLPprocess.function_normalization(cleansing)
	negation = NLPprocess.function_negation_tweet(normalization)
	tokenize = NLPprocess.function_tokenize_tweet(negation)
	stopword = NLPprocess.funtion_stopword_tweet(tokenize)
	pos_tagging = NLPprocess.function_pos_tagging(stopword)
	stemmer = NLPprocess.function_stemmer_tweet(pos_tagging)
	lexicon_based = NLPprocess.funtion_lexicon_based(stemmer)
	naive_bayes = NLPprocess.function_naive_bayes(lexicon_based)
	information_classifier = NLPprocess.function_information_classifier(naive_bayes)

	return render_template('prabowo.html', crawling = data_testing, caseHolding = caseholding, cleansing = cleansing, normalization = normalization, negation = negation, tokenize = tokenize, stopwords = stopword, pos = pos_tagging, stemming = stemmer, lexicon = lexicon_based, naive = naive_bayes, information = information_classifier)
