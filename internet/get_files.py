import os
from datetime import datetime, date

os.system("mkdir -p anon")

# Get the raw files for these days
days = (1,) # 5, 25)

# Base of the CAIDA telescope URL
base = "data.caida.org/datasets/security/telescope-educational/exercises/anon/"

# The filenames have this pattern
template = "ucsd-nt.anon.XXXXXXXXXX.flowtuple.cors.gz"


def get_day(day):
    bdate = date(2012, 4, day)
    for hour in range(0, 24):
        btime = datetime(2012, 4, day, hour)
        udt = (btime - datetime(1970, 1, 1)).total_seconds()
        fname = template.replace("XXXXXXXXXX", str(int(udt)))
        url = "%s%s" % (base, fname)
        os.system("wget http://%s/%s -O anon/%s/%s" % (base, fname, bdate.isoformat(), fname))


for day in days:
    bdate = date(2012, 4, day)
    os.system("mkdir -p anon/%s" % bdate.isoformat())
    get_day(day)
