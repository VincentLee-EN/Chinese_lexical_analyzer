from flask import Flask, request, abort
from cws.segmenter import BiLSTMSegmenter


app = Flask(__name__)

@app.route('/cws', methods=['POST'])
def segment():
    if not request.json:
        abort(400)
    text = request.json['text']
    if not text or text == '':
        abort(400)
    return segmenter.predict(text)


if __name__ == '__main__':
    segmenter = BiLSTMSegmenter(data_path='data/your_dict.pkl', model_path='checkpoints/cws.ckpt/')
#    print('示例2：', segmenter.predict('我和林长开的通话记录'))
    segmenter.predict('2003年10月15日，杨利伟乘由长征二号F火箭运载的神舟五号飞船首次进入太空，象征着中国太空事业向前迈进一大步，起到了里程碑的作用。')
    # app.run(host='0.0.0.0', port=7777, debug=True, use_reloader=False)
