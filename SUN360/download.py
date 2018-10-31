import os
import urllib2

path = './SUN360_urls_9104x4552'
save_path = './SUN360_9104x4552'

for root, dirs, files in os.walk(path):
        for file in files:
                with open(root + '/' + file) as f1:
                        for line in f1:
                                if not dirs:
                                        dir = save_path + '/' + root.split('/')[-1] + '/' + file.split('.')[0]
                                else:
                                        dir = save_path + '/' + file.split('.')[0]

                                if not os.path.exists(dir):
                                        os.makedirs(dir)

                                try:
                                        filedata = urllib2.urlopen(line[:-1])
                                        datatowrite = filedata.read()
                                        with open(dir + '/' + line.split('/')[-1][:-1], 'wb') as f2:
                                                f2.write(datatowrite)
                                        print(line + 'Downloaded!')
                                except urllib2.HTTPError:
                                        print(line + 'Not Found!')

        #print(root, dirs, files)