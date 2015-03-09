import json
print('JSON Test')

data = [1,2,3,4,5,6]

with open('data.json','w') as f:
    json.dump(data,f)
    f.close()

with open('data.json','r') as f:
    print(f.read())
    f.close()


