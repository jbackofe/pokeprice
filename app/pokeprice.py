# This example requires the 'message_content' privileged intent to function.

import discord
import requests

import pandas as pd
import sys
import math
import json
from functools import reduce
import dataframe_image as dfi
import nltk
nltk.download('punkt')

sys.path.insert(0, './classifier')
from similarity_inf import load_model, load_class_info, import_inf_data

data = json.load(open('keys.json'))

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
 
    return str1

def card_prices(df):
    return df[['name',
           'name_set',
           'number',
           'price_average',
           'price_avg1',
           'price_avg7',
           'price_avg30',
           'price_pctDiff_30',
           'dt']].sort_values('price_average', ascending=False).reset_index(drop=True)

# query: name of a card
def search_card_name(df, query):
    query = query.upper()
    query = nltk.word_tokenize(query)

    def find_values(df, query):
        dfs = []
        for val in query:
            df = df[df['card_set'].str.contains(val, case=False, na=False, regex=True)]
            dfs.append(df)
        return reduce(lambda x, y: pd.merge(x, y, how='inner', on=['card_set'], suffixes=(None, '_y')), dfs)

    df['name'] = df['name'].str.upper()
    df['name_set'] = df['name_set'].str.upper()
    df['card_set'] = df['name'].astype(str) + ' ' + df['name_set'].astype(str) + ' ' + df['number'].astype(str)
    df = find_values(df, query)
    df = df[['name',
           'name_set',
           'card_set',
           'number',
           'price_average',
           'price_avg1',
           'price_avg7',
           'price_avg30',
           'price_pctDiff_30',
           'dt']]

    df = df.drop(['card_set'], axis=1).sort_values(['price_average'], ascending=False)
    return df

# query: name of a card
def search_below_hundred(df, query):
    query = query.upper()
    query = nltk.word_tokenize(query)

    def find_values(df, query):
        dfs = []
        for val in query:
            df = df[df['card_set'].str.contains(val, case=False, na=False, regex=True)]
            dfs.append(df)
        return reduce(lambda x, y: pd.merge(x, y, how='inner', on=['card_set'], suffixes=(None, '_y')), dfs)

    df = df[df['price_average'] < 100]
    df = df[df['price_average'] > 20]
    df['name'] = df['name'].str.upper()
    df['name_set'] = df['name_set'].str.upper()
    df['card_set'] = df['name'].astype(str) + ' ' + df['name_set'].astype(str) + ' ' + df['number'].astype(str)
    df = find_values(df, query)
    df = df[['name',
           'name_set',
           'card_set',
           'number',
           'price_average',
           'price_avg1',
           'price_avg7',
           'price_avg30',
           'price_pctDiff_30',
           'dt']]

    df = df.drop(['card_set'], axis=1).sort_values(['price_average'], ascending=False)
    return df

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        # hello
        if message.content.startswith('!hello'):
            print(message.content)
            await message.reply('Hello!', mention_author=True)

        # searches across card name, set name and card number
        if message.content.startswith('!search'):
            query = message.content
            print(query[8:])
            df = search_card_name(cards, query[8:])

            pages = math.ceil(len(df)/20)
            print('pages=', pages)
            a = 0
            b = 0
            for page in range(pages):
                print(page)
                a = b
                b = (page + 1) * 20
                print('a', a, 'b', b)
                df_page = df[a:b]

                dfi.export(df_page, "plot.png")
                file=discord.File('plot.png')
                e = discord.Embed()
                e.set_image(url="attachment://plot.png")
                await message.reply(file = file, embed = e, mention_author=True)

        # searches cards between 20-100 average price
        if message.content.startswith('!100'):
            query = message.content
            print(query[5:])
            df = search_below_hundred(cards, query[5:])
            # df_page = df[:20]

            pages = math.ceil(len(df)/20)
            print('pages=', pages)
            a = 0
            b = 0
            for page in range(pages):
                print(page)
                a = b
                b = (page + 1) * 20
                print('a', a, 'b', b)
                df_page = df[a:b]

                dfi.export(df_page, "plot.png")
                file=discord.File('plot.png')
                e = discord.Embed()
                e.set_image(url="attachment://plot.png")
                await message.reply(file = file, embed = e, mention_author=True)

        # Returns top 20 cards by average price
        if message.content.startswith('!cards'):
            df = card_prices(cards)[:20]
            dfi.export(df, "plot.png")
            file=discord.File('plot.png')
            e = discord.Embed()
            e.set_image(url="attachment://plot.png")
            await message.reply(file = file, embed = e, mention_author=True)

        # Classifies the attached image
        if message.content.startswith('!find'):
            e = discord.Embed()
            e.set_image(url=message.attachments[-1])
            image_url = e.image.url
            print(image_url)
            img_data = requests.get(image_url).content
            with open('./images/collected/image.jpg', 'wb') as handler:
                handler.write(img_data)

            # Classify card
            x_inf = import_inf_data('./images/collected/', (120, 185))
            pred = reloaded_model.match(x_inf, cutpoint="optimal", no_match_label=len(labels)-1)

            # Get the predicted class name
            outputs = []
            for val in pred:
                if val > len(class_names)-1:
                    outputs.append('Unknown')
                else:
                    outputs.append(class_names[val])
            
            await message.reply(listToString(outputs), mention_author=True)

# Load card price table
cards = pd.read_csv('./price_tables/cards.csv')

# Load classifier model
reloaded_model = load_model('./models/pokemon_similarity')
labels, class_names = load_class_info()

intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(data['pokeprice_token'])
