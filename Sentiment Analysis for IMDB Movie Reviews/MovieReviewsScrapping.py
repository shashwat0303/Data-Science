from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import requests

posReviews = []
negReviews = []

def PositiveReviews(url):
    page = requests.get(url).text.encode('ascii', 'ignore')
    soup = bs(page, 'html.parser')
    div = soup.find('div', {'id' : 'tn15content'})
    p = div.findAll('p', {'b' : None})
    reqPara = [para for para in p[:-1] if str(para.text.encode('ascii', 'ignore')) != "*** This review may contain spoilers ***"]
    for para in reqPara:
        posReviews.append(str(para.text.encode('ascii', 'ignore')))
    return posReviews

def NegativeReviews(url):
    page = requests.get(url).text.encode('ascii', 'ignore')
    soup = bs(page, 'html.parser')
    div = soup.find('div', {'id' : 'tn15content'})
    p = div.findAll('p', {'b' : None})
    reqPara = [para for para in p[:-1] if str(para.text.encode('ascii', 'ignore')) != "*** This review may contain spoilers ***"]
    for para in reqPara:
        negReviews.append(str(para.text.encode('ascii', 'ignore')))
    return negReviews

def NextPage(url):
    page = requests.get(url).text.encode('ascii', 'ignore')
    soup = bs(page, 'html.parser')
    div = soup.find('div', {'id' : 'tn15content'})
    table = div.findAll('table')
    tds = table[1].findAll('td', {'nowrap' : "1"})
    a = tds[-1].findAll('a')
    if len(a) < 9:
        return None
    return "http://www.imdb.com/title/tt0241527/" + a[-1].get('href')

posURL = 'http://www.imdb.com/title/tt0120338/reviews?count=498&filter=love;filter=love;start=0'
negURL = 'http://www.imdb.com/title/tt0120338/reviews?count=346&filter=hate;filter=hate;start=0'

PositiveReviews(posURL)
NegativeReviews(negURL)

with open('PositiveReviews.txt', 'a') as f:
    for review in posReviews:
        f.write(review + "\t\t\t")

with open('NegativeReviews.txt', 'a') as f:
    for review in negReviews:
        f.write(review + "\t\t\t")

'''reviews = posReviews
reviewType = np.ones(len(posReviews))
reviewType = reviewType.tolist()

for review in negReviews:
    reviews.append(review)
    reviewType.append(0)

allReviews = {}
allReviews['reviews'] = reviews
allReviews['type'] = reviewType

df = pd.DataFrame(allReviews)
df.to_csv('MovieReviews.csv')'''
