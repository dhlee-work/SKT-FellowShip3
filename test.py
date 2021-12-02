import pandas as pd
import os
os.getcwd()

import glob
glob.glob('*')
print('aaaaaaaaaaa')



from googletrans import Translator
translator = Translator()
result = translator.translate('안녕하세요.', dest="en")
print(result[0].text)




dic = {}
dic['en'] = set()
dic['en'].update({1, 2, 3, 7})
dic
