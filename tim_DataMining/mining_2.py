#%%
import parsel
import requests

#数据获取
url = 'https://gz.fang.lianjia.com/loupan/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    print(response.text)
except requests.exceptions.RequestException as e:
    print(f"请求出错: {e}")

print(response.text)

#解析提取数据
selector = parsel.Selector(response.text)
lis = selector.css('.resblock-list-container .resblock-list-wrapper .resblock-list')

for li in lis:
    title = li.css('.resblock-list-container .resblock-list-wrapper .resblock-list .resblock-img-wrapper::text').get() #名称
    types = li.css('.resblock-desc-wrapper .resblock-name span::text').get() #销售类型
    average_price = li.css('.resblock-desc-wrapper .resblock-price .main-price .number ::text').get() #均价
    total_price = li.css('.resblock-desc-wrapper .resblock-price .second ::text').get() #总价
    area = li.css('.resblock-desc-wrapper .resblock-price .second ::text').get() #地区
    location = li.css('.resblock-desc-wrapper .resblock-price .second ::text').get() #位置
    street = li.css('.resblock-desc-wrapper .resblock-price .second ::text').get() #具体位置
    structure = li.css('.resblock-desc-wrapper .resblock-room span:hover ::text').get() #室厅数
    dimensions = li.css('.resblock-desc-wrapper .resblock-area span ::text').get() #面积
    tags = li.css('.resblock-desc-wrapper .resblock-tag span ::text').get() #标签
    dit = {title,
           types,
           average_price,
           total_price,
           area,
           location,
           street,
           structure,
           dimensions,
           tags,
           }

    print(dit)
#%%

#%%
