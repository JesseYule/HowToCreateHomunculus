from LAC import LAC
import pandas as pd

lac = LAC()

# 这里为了简化就不连接数据库分析了
data = pd.read_excel('data.xlsx')

word_list = ['温度', '天气']
content = {'loc': None, 'time': None}

turn = 0

while True:

    # 获取用户输入，进行命名实体识别
    word = input('输入： ')
    result = lac.run(word)

    # 意图识别
    # 这里为了简化通过特征词和规则分析意图，可利用深度学习模型对句子进行分类实现准确度更高的意图识别
    # 另一方面，真正的难点应该是如何对多轮对话进行综合的意图识别，比如第一轮客户问天气，符合系统需求，但是客户只提供了城市信息，缺乏了日期
    # 所以在第二轮，客户就需要提供日期信息，可是这时候如果单独对该部分输入进行意图识别，分析就会出错

    if turn == 0 and word_list[0] not in result[0] and word_list[1] not in result[0]:
        # 这里简化了流程，只对最初的输入进行意图识别
        print('对不起，本系统只提供天气查询功能')
    else:
        turn = 1

        # 通过content记录每轮用户输入的关键信息，也就是对话管理这个模块负责的内容

        if 'TIME' in result[1]:
            index = result[1].index('TIME')
            content['time'] = str(result[0][index])
        else:
            print('请问您想知道具体哪一天的天气')

        if 'LOC' in result[1]:
            index = result[1].index('LOC')
            content['loc'] = str(result[0][index])
        else:
            print('请问您想知道那座城市的天气')

        if content['loc'] and content['time']:
            try:
                # 查询语句，可改成SQL或者Cypher语句，可结合意图识别的结果，分类查询
                output = data[(data['城市'] == content['loc']) & (data['日期'] == content['time'])]['温度'].values[0]
                print(content['loc'] + content['time'] + '的温度是' + str(output) + '度')

            except Exception as e:
                print('对不起，本系统只提供广州、上海、北京9月1号到9月13号的数据')

            # 完成一次搜索，清空记录的信息
            content['loc'] = None
            content['time'] = None
