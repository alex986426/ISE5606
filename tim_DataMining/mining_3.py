# 导入数据请求模块 第三方模块, 需要安装
import requests
# 导入数据解析模块 第三方模块, 需要安装
import parsel
# 导入csv
import csv

f = open('new0.csv', mode='w', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(f, fieldnames=[
    '标题',
    '小区',
    '区域',
    '总价',
    '单价',
    '户型',
    '面积',
    '朝向',
    '装修',
    '楼层',
    '年份',
    '建筑结构',
    '详情页',
])
csv_writer.writeheader()
"""
1. 发送请求, 模拟浏览器对于url地址发送请求

"""
# 模拟浏览器
headers = {
    # 用户代理 表示浏览器基本身份信息
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
}
for page in range(1, 101):
    # 请求链接:
    url = f'https://cs.lianjia.com/ershoufang/pg{page}/'
    # 发送请求
    response = requests.get(url=url, headers=headers)
    """
    2. 获取数据, 获取服务器返回响应数据  <获取整个网页的数据内容>
 
    
    3. 解析数据, 提取我们需要的数据内容  <获取其中一部分内容>
        数据: 房源基本信息
    
    css选择器: 根据标签属性提取数据内容
        - 获取所有房源所在li标签
    """
    # 把获取下来的网页数据, 转成可解析对象 <Selector xpath=None data='<html class=""><head><meta http-equiv...'>
    selector = parsel.Selector(response.text)  # 选择器对象
    # 获取所有房源所在li标签
    lis = selector.css('.sellListContent li .info')
    # for循环遍历
    for li in lis:
        """
        提取具体房源信息
        """
        title = li.css('.title a::text').get()  # 标题
        area_info = li.css('.positionInfo a::text').getall()  # 区域信息
        area_1 = area_info[0]  # 小区
        area_2 = area_info[1]  # 区域
        totalPrice = li.css('.totalPrice span::text').get()  # 总价
        unitPrice = li.css('.unitPrice span::text').get().replace('元/平', '')  # 单价
        houseInfo = li.css('.houseInfo::text').get().split(' | ')  # 房源信息
        if len(houseInfo) == 7:
            date = houseInfo[5]
        else:
            date = '未知'
        HouseType = houseInfo[0]  # 户型
        HouseArea = houseInfo[1].replace('平米', '')  # 面积
        HouseFace = houseInfo[2]  # 朝向
        HouseInfo_1 = houseInfo[3]  # 装修
        fool = houseInfo[4]  # 楼层
        HouseInfo_2 = houseInfo[-1]  # 建筑结构
        href = li.css('.title a::attr(href)').get()  # 详情页
        dit = {
            '标题': title,
            '小区': area_1,
            '区域': area_2,
            '总价': totalPrice,
            '单价': unitPrice,
            '户型': HouseType,
            '面积': HouseArea,
            '朝向': HouseFace,
            '装修': HouseInfo_1,
            '楼层': fool,
            '年份': date,
            '建筑结构': HouseInfo_2,
            '详情页': href,
        }
        csv_writer.writerow(dit)
        print(dit)
