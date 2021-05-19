import requests
import json

def postScoresToDatabase():
    host = '172.16.0.4'
    port = 8080
    notesUrl = f'http://{host}:{port}/note'
    headers = {'content-type': 'application/json'}
    with open('C:/Users/joeba/Documents/github/ds-central/test.json', 'r') as rf:
        jsonString = json.load(rf)
    #jsonString = json.dumps(jsonString)
    try:
        #ret = requests.post(notesUrl, data=jsonString, headers=headers, verify=False)
        ret = requests.post(notesUrl, json=jsonString)
        print(ret.json())
        return ret.json()['noteId']
    except requests.exceptions.RequestException as e:
        print(e)

postScoresToDatabase()