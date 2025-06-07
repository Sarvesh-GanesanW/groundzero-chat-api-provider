from utils.credProvider import getCredentials
import psycopg2
import psycopg2.extras

def executeFunction(databaseName, function, params):
    result = {}
    try:
        credential = getCredentials()
        connection = psycopg2.connect(user=credential['username'],password=credential['password'],host=credential['host'], database=databaseName, port=credential['port'])
        cursor = connection.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
        cursor.callproc(function, params)
        result = cursor.fetchall()
        connection.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while connecting to PostgreSQL: ", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
    return result
