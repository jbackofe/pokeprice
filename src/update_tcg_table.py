import pandas as pd
from datetime import date
import json

from pokemontcgsdk import Card
from pokemontcgsdk import Set
from pokemontcgsdk import RestClient

f = open('keys.json')
data = json.load(f)
RestClient.configure(data['tcg_api_token'])

def parse_averageSellPrice(row):
    if (row != None) and (row['prices'] != None) and (row['prices']['averageSellPrice'] != None):
        return row['prices']['averageSellPrice']
    else:
        return None
    
def parse_lowPrice(row):
    if (row != None) and (row['prices'] != None) and (row['prices']['lowPrice'] != None):
        return row['prices']['lowPrice']
    else:
        return None
    
def parse_avg1(row):
    if (row != None) and (row['prices'] != None) and (row['prices']['avg1'] != None):
        return row['prices']['avg1']
    else:
        return None
    
def parse_avg7(row):
    if (row != None) and (row['prices'] != None) and (row['prices']['avg7'] != None):
        return row['prices']['avg7']
    else:
        return None
    
def parse_avg30(row):
    if (row != None) and (row['prices'] != None) and (row['prices']['avg30'] != None):
        return row['prices']['avg30']
    else:
        return None
    
def parse_reverseHoloAvg1(row):
    if (row != None) and (row['prices'] != None) and (row['prices']['reverseHoloAvg1'] != None):
        return row['prices']['reverseHoloAvg1']
    else:
        return None
    
def parse_reverseHoloAvg7(row):
    if (row != None) and (row['prices'] != None) and (row['prices']['reverseHoloAvg7'] != None):
        return row['prices']['reverseHoloAvg7']
    else:
        return None
    
def parse_reverseHoloAvg30(row):
    if (row != None) and (row['prices'] != None) and (row['prices']['reverseHoloAvg30'] != None):
        return row['prices']['reverseHoloAvg30']
    else:
        return None

def update_tcg_table():
    cards = pd.DataFrame(Card.all())
    sets = pd.DataFrame(Set.all())

    cards = cards[['name', 'cardmarket', 'id', 'images', 'subtypes', 'supertype', 'tcgplayer']]
    cards['set'] = cards['id'].str.split('-', expand=True).rename(columns={0: 'set', 1: 'number'})['set']
    cards['number'] = cards['id'].str.split('-', expand=True).rename(columns={0: 'set', 1: 'number'})['number']
    cards = pd.merge(left=cards, right=sets, how='left', left_on='set', right_on='id', suffixes=(None, '_set')).drop('id_set', axis=1)

    cards['price_average'] = cards.apply(lambda x: parse_averageSellPrice(x['cardmarket']), axis=1)
    cards['price_low'] = cards.apply(lambda x: parse_lowPrice(x['cardmarket']), axis=1)
    cards['price_avg1'] = cards.apply(lambda x: parse_avg1(x['cardmarket']), axis=1)
    cards['price_avg7'] = cards.apply(lambda x: parse_avg7(x['cardmarket']), axis=1)
    cards['price_avg30'] = cards.apply(lambda x: parse_avg30(x['cardmarket']), axis=1)
    cards['price_reverseHoloAvg1'] = cards.apply(lambda x: parse_reverseHoloAvg1(x['cardmarket']), axis=1)
    cards['price_reverseHoloAvg7'] = cards.apply(lambda x: parse_reverseHoloAvg1(x['cardmarket']), axis=1)
    cards['price_reverseHoloAvg30'] = cards.apply(lambda x: parse_reverseHoloAvg1(x['cardmarket']), axis=1)

    cards['price_pctDiff_30'] = ((cards['price_avg1'] - cards['price_avg30'])/((cards['price_avg1'] + cards['price_avg30'])/2))*100
    cards['price_pctDiff_7'] = ((cards['price_avg1'] - cards['price_avg7'])/((cards['price_avg1'] + cards['price_avg7'])/2))*100

    cards['dt'] = date.today()

    # save to csv
    cards.to_csv('price_data/cards_2022_12_06.csv')

if __name__ == "__main__":
    print('Updating tcg table...')
    update_tcg_table()
    print('Updated tcg table!')