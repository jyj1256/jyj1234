import os
from openai import OpenAI
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import pinecone
import datetime

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, model='text-embedding-ada-002'):
    client = OpenAI(api_key=openai_api_key)
    # content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    # response = client.embeddings.create(input=content, model=model)
    # vector = response['data'][0]['embedding']  # this is a normal list
    content = content.replace("\n", " ")
    vector = client.embeddings.create(input = [content], model=model).data[0].embedding
    return vector


#gpt-3.5-turbo-instruct
#
def gpt3_completion(prompt, model='ft:gpt-3.5-turbo-0613:personal::8uaHYqST'):
    max_retry = 5
    retry = 0
    # prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = client.chat.completions.create(
                model = model,
                messages=[
                    {"role": "system", "content": "날짜 체계를 고려해서 두번째 문장에대한 대답해"},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def load_conversation(results):
    result = list()
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        result.append(info)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()


if __name__ == '__main__':
    convo_length = 30
    openai_api_key = open_file('key_openai.txt')
    client = OpenAI(api_key=openai_api_key)
    pinecone.init(api_key=open_file('key_pinecone.txt'), environment='gcp-starter')
    # pinecone.create_index("raven-mvp", dimension=1586, metric="cosine") #생성
    # print(pinecone.describe_index("raven-mvp")) #설명
    vdb = pinecone.Index("raven-mvp")
    while True:
        now = datetime.datetime.now()
        formatted_date_time = now.strftime("%Y%m%d-%H:%M")
        hangletime = str(formatted_date_time)
        print(type(formatted_date_time))
        #### get user input, save it, vectorize it, save to pinecone
        payload = list()
        a = input('\n\nUSER: ')
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('USER', timestring, a)
        message = hangletime + " : " + a
        print(message)
        vector = gpt3_embedding(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'USER', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        #### search for relevant messages, and generate a response
        results = vdb.query(vector=vector, top_k=convo_length) #search
        conversation = load_conversation(results)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id'
        prompt = a
        #### generate response, vectorize, save, etc
        output = gpt3_completion(prompt)
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('RAVEN', timestring, output)
        message = output
        vector = gpt3_embedding(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'RAVEN', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        vdb.upsert(payload)
        print('\n\nRAVEN: %s' % output)