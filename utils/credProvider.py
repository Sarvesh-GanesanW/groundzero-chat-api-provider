import boto3
import os
import json
from appCache import app_cache
from constants import Constants

def getCredentials():
    credential = app_cache.get(Constants.DB_CREDENTIALS)
    if credential:
        return credential
    else:
        session = boto3.session.Session()
        client = session.client(
            service_name="secretsmanager", region_name=os.environ["REGION"]
        )
        get_secret_value_response = client.get_secret_value(
            SecretId=os.environ["DB_SECRET_NAME"]
        )
        credential = json.loads(get_secret_value_response["SecretString"])
        app_cache.set(Constants.DB_CREDENTIALS, credential)
        
        return credential
