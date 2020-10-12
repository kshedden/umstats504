import os
from datetime import datetime, date

os.system("mkdir -p results")

# Process these days
days = (1, 5, 25)

# The filenames have this pattern
template = "ucsd-nt.anon.XXXXXXXXXX.flowtuple.cors.gz"

def process_day(day):
    bdate = date(2012, 4, day)
    fnames = []
    for hour in range(0, 24):
        btime = datetime(2012, 4, day, hour)
        udt = (btime - datetime(1970, 1, 1)).total_seconds()
        fname = template.replace("XXXXXXXXXX", str(int(udt)))
        fnames.append(fname)
    return fnames

for day in days:
    bdate = date(2012, 4, day)
    fnames = process_day(day)
    fid = open("telescope_files.txt", "w")
    for fname in fnames:
        fid.write(fname + "\n")
    fid.close()
    d = bdate.isoformat()
    cmd = "cat telescope_files.txt | rush ./unique anon/%s/{} results/%s" % (d, d)
    os.system(cmd)

