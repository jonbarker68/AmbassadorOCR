import urllib2
import json

#BASE_URL = 'http://localhost:5000'
BASE_URL = 'http://178.62.73.32'


def submit(username, password, trialname, trialdata):

    jdata = json.dumps({"username": username, "password": password,
                       "trialname": trialname,
                       "trialdata": trialdata.tolist()})

    request = urllib2.Request('{}/api_submit'.format(BASE_URL),
                              jdata, {'Content-Type': 'application/json'})
    f = urllib2.urlopen(request)
    response = f.read()
    f.close()
