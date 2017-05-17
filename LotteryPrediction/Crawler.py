# -*- encoding:utf-8 -*-
import urllib.request
import urllib
import re
from bs4 import BeautifulSoup


def getPage(href):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.2) AppleWebKit/525.13 (KHTML, like Gecko) Chrome/0.2.149.27'

    headers = {
        'User-Agent': user_agent}

    req = urllib.request.Request(href, headers=headers)
    post = urllib.request.urlopen(req)
    return post.read()


def getPageNum(url):
    num = 0
    page = getPage(url)
    soup = BeautifulSoup(page)
    strong = soup.find('td', colspan='7')
    # print()
    if strong:
        result = strong.get_text().split(' ')
        list_num = re.findall("[0-9]{1}", result[1])
        for i in range(len(list_num)):
            num = num * 10 + int(list_num[i])
        return num
    else:
        return 0


def getText(url):

    for list_num in range(1, getPageNum(url)):
        print(list_num)
        href = 'http://kaijiang.zhcw.com/zhcw/html/ssq/list_' + \
            str(list_num) + '.html'
        page = BeautifulSoup(getPage(href))
        em_list = page.find_all('em')
        div_list = page.find_all('td', {'align': 'center'})

        n = 0
        fp = open("num.txt", "w")
        for div in em_list:
            text = div.get_text()
#             text = text.encode('utf-8')
            n = n + 1
            if n == 7:
                text = text + '\n'
                n = 0
            else:
                text = text + ','
            fp.write(str(text))
        fp.close()


url = "http://kaijiang.zhcw.com/zhcw/html/ssq/list_1.html"
getText(url)
