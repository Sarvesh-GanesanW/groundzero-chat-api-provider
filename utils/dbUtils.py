import os
from appCache import app_cache
from utils.dbProvider import executeFunction
site = os.getenv("ENV")
defaultDB = "groundzero" + site if site else ""

def getSiteDB(siteName):
    siteDBPrefix = "DBNameFor_"
    siteDB = app_cache.get(siteDBPrefix + siteName)
    if siteDB:
        return siteDB
    else:
        sitesInDb = executeFunction(defaultDB, "fn_r_sites", [])
        for site in sitesInDb:
            app_cache.set(siteDBPrefix+site["Name"], site["DBName"])
        return app_cache.get(siteDBPrefix + siteName)