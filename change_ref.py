fref = open("C:\\Users\\Michael\\Documents\\study\\Thesis\\3 - papers\\Thesis\\ref list.txt", "r")
refs = []
for line in fref:
    fields = line.split(" ")
    refs += [(fields[0].replace("\n", "").replace("\r", ""), fields[1].replace("\n", "").replace("\r", ""))]
with open("C:\\Users\\Michael\\Desktop\\m.txt", 'r') as myfile:
    data = myfile.read()
for ref in refs:
    data = data.replace(ref[0], ref[1])

fw = open("C:\\Users\\Michael\\Desktop\\m2.txt", "w")
fw.write(data)
