# This example requires the 'message_content' privileged intent to function.

import discord

import pandas as pd
import json
from functools import reduce
import dataframe_image as dfi
import nltk
nltk.download('punkt')

data = json.load(open('keys.json'))

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

cards = pd.read_csv('price_data/cards_2022_12_06.csv')

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        if message.content.startswith('!hello'):
            print(message.content)
            await message.reply('Hello!', mention_author=True)

        if message.content.startswith('!imgay'):
            print(message.content)
            await message.reply('urgay!', mention_author=True)

        if message.content.startswith('!search'):
            query = message.content
            print(query[8:])
            df = search_card_name(cards, query[8:])[:20]
            dfi.export(df, "plot.png")
            file=discord.File('plot.png')
            e = discord.Embed()
            e.set_image(url="attachment://plot.png")
            await message.reply(file = file, embed = e, mention_author=True)

        if message.content.startswith('!cards'):
            df = card_prices(cards)[:20]
            dfi.export(df, "plot.png")
            file=discord.File('plot.png')
            e = discord.Embed()
            e.set_image(url="attachment://plot.png")
            await message.reply(file = file, embed = e, mention_author=True)

        if message.content.startswith('!grade'):
            # e = discord.Embed()
            # e.set_image(url=message.attachments[-1])
            # await message.reply(embed=e)
            await message.reply('10/10', mention_author=True)

intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run(data['pokeprice_token'])
