from mangum import Mangum
from app import app
mangum_adapter = Mangum(app)

def handler(event, context):
    print("Received AWS Lambda event:", event)
    print("Received AWS Lambda context:", context)
    response = mangum_adapter(event, context)

    return response