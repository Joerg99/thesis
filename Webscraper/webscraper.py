'''
Created on 19 Oct 2018

@author: Jorg
'''

from bs4 import BeautifulSoup
import urllib3
import requests
import csv

def scrape_and_export_lyrikmond():
    http = urllib3.PoolManager()
    
    headers = requests.utils.default_headers()
    headers.update({"User-Agent":
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"})
    poems = []
    for i in range(1,2500):
        url = 'https://www.lyrikmond.de/gedichte/gedichttext-neu.php?g='+str(i)
        response = http.request('GET', url, headers)
        soup = BeautifulSoup(response.data)
        poem = [str(soup.body.p)]
        if poem[0][3:8] != "<br/>":
            poems.append(poem)
    for i in range(len(poems)):
        temp = str(poems[i])
        poems[i] = str.encode(temp)

    with open('poems_lyrikmond.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(poems)
        
def scrape_and_export_mumag():
    # ********************************* scrape
    http = urllib3.PoolManager()
    headers = requests.utils.default_headers()
    headers.update({"User-Agent":
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"})
    url = 'https://www.mumag.de/gedichte/'
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data)
    links=[]

    for link in soup.find_all('a'):
        links.append(link.get('href'))
    # ********************************* generate all links
    poem_links = []
    for link in links:
        if isinstance(link, str):
            if link[-5:] == ".html" and '_' in link and link[:5] != "thema":
                poem_links.append(url+link)
    # ********************************* generate all links
    poems = []
    for link in range(1,5):
        response = http.request('GET', poem_links[link])
        soup = BeautifulSoup(response.data)
        headline, author, text= soup.h2, soup.h3, soup.find_all('p')[:-2]
        poem = str(headline), str(author), str(text)
        poems.append([poem])
    print(len(poems))
    
    with open('poems_mumag.csv', 'w', encoding='utf-8') as file:
        #for element in poems:
        #    file.write(element)
        writer = csv.writer(file)
        for element in poems:
            writer.writerow(element)
    

def load_poems_from_csv(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        poems_in= list(reader)
    return poems_in

if __name__ == '__main__':
    scrape_and_export_mumag()

    poems = load_poems_from_csv('poems_mumag.csv')
    print(len(poems))
    
    