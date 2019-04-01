# encoding=UTF-8

from flask import Flask, request, abort
from cws.segmenter import BiLSTMSegmenter
from summary import textrank
from summary import words_merge

app = Flask(__name__)


@app.route('/cws', methods=['POST'])
def segment():
    if not request.json:
        abort(400)
    text = request.json['text']
    if not text or text == '':
        abort(400)
    return segmenter.predict(text)


def keywords(doc, limit=5, merge=False):
    rank = textrank.KeywordTextRank(doc)
    rank.solve()
    ret = []
    for w in rank.top_index(limit):
        ret.append(w)
    if merge:
        wm = words_merge.SimpleMerge(doc, ret)
        return wm.merge()
    return ret


if __name__ == '__main__':
    segmenter = BiLSTMSegmenter(data_path='data/your_dict.pkl', model_path='checkpoints/cws.ckpt/')
    #    print('示例2：', segmenter.predict('我和林长开的通话记录'))
    texts = [
        '长飞公司创建于1988年5月，由中国电信集团公司、荷兰德拉克通信科技公司、武汉长江通信集团股份有限公司共同'
        '投资。公司总部位于武汉市东湖高新技术开发区关山二路四号，占地面积达十七万平方米，是当今中国产品规格'
        '最齐备、生产技术最先进、生产规模最大的光纤光缆产品以及制造装备的研发和生产基地。 '
        '长飞有限公司的光纤光缆产品及多种网络建设解决方案能够满足每一个行业用户的不同需求，'
        '已广泛应用于中国电信、中国移动、中国联通等通信运营商，以及电力、广电、交通、教育、国防、航天、'
        '化工、石油、医疗等行业领域，并远销美国、日本、韩国、台湾、东南亚、中东、非洲等50多个国家和地区。'
        ,
        "2003年10月15日，杨利伟乘由长征二号F火箭运载的神舟五号飞船首次进入太空，"
        "象征着中国太空事业向前迈进一大步，起到了里程碑的作用。')"
    ]
    for sentence in texts:
        print('======')
        print(sentence)
        words, tags = segmenter.predict(sentence)
        for i in range(len(words)):
            print(words[i]+ '/'+ tags[i], end='  ')
        print()
