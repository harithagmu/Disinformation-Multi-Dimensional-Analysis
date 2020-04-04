import requests
from bs4 import BeautifulSoup
import pandas as pd
import os 
from urllib.request import urlopen, Request

os.chdir("path")
buzzdata = pd.read_csv("facebook-fact-check.csv")

print(buzzdata.shape)
buzzdata = buzzdata[buzzdata['Post Type'] == 'link']

def request_until_succeed(url):
    req = Request(url)
    success = False
    if success is False:
        try:
            response = urlopen(req)
            if response.getcode() == 200:
                success = True
                soup = BeautifulSoup(response, 'lxml')
                print(url)#soup.prettify())
                print(soup.find('div', class_="userContent"))
                
            else:
                print("request returned ", response.getcode())
        except Exception as e:
            print(e)
            print("Error for URL {}: {}".format(url, datetime.datetime.now()))
            print("Retrying.")

    return soup


for link in buzzdata['Post URL'].head(1):
    request_until_succeed(link)
	buzzdata['articletext'] = soup.find('p')
	buzzdata['articleslice'] = soup.title.text
    
buzzdata.to_csv("data.csv")    
    
    
    
