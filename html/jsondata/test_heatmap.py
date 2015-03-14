import random, os, json


def writeJSON(array, size, filename):
    valuesdict = {'values': array[-size:]}
    with open(os.path.join(filename),'wb') as f:
        f.write(json.dumps(valuesdict))

rms_heatmap_list = []
print(rms_heatmap_list)

for day_number in xrange(50):
    for ten_minute_number in xrange(144):
        rms_heatmap_list.append([day_number, ten_minute_number, random.randint(220,240)])

print(str(rms_heatmap_list))
writeJSON(rms_heatmap_list, 14400, 'rmsHeatmap.json')

