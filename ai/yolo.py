from ultralytics import YOLO
from collections import OrderedDict

def process_img(img):
    model = YOLO('ai/best.pt')
    result = model(img)[0].boxes

    cls = result.cls.tolist()
    xywhn = result.xywhn.tolist()
    dic = {0: '그릇', 1: '쌀밥', 2: '기타잡곡밥', 3: '콩밥', 4: '보리밥',
           5: '돌솥밥', 6: '현미밥', 7: '흑미밥', 8: '유부초밥', 9: '참치김밥', 10: '떡라면',
           11: '라면', 12: '비빔국수', 13: '쌀국수', 14: '오일소스스파게티', 15: '쫄면',
           16: '크림소스스파게티', 17: '해물칼국수', 18: '떡국', 19: '달걀국', 20: '미역 국',
           21: '배추된장국', 22: '콩나물국', 23: '떡갈비', 24: '치킨데리야끼', 25: '치킨윙', 26: '두부구이', 27: '달걀말이', 28: '멸치볶음', 29: '어묵볶음',30: '소세지볶음',
           31: '순대볶음', 32: '떡볶이', 33: '깍두기', 34: '깻잎김치', 35: '동치미', 36: '배추김치', 37: '백김치', 38: '열무김치', 39: '오이소박이', 40: '총각김치'}
    
    #cls를 dic에 있는 이름형식으로 변경
    name = [dic[key] for key in cls]

    #name:좌표로 합치기
    ans = [[n] + xy for n, xy in zip(name, xywhn)]
    #그릇 제거
    ans = [item for item in ans if '그릇' not in item[0]]

    send = []
    
    for item in ans:
        food_dic = {
            "foodName": item[0],
            "XYCoordinate": [item[1:]]
        }
        send.append(food_dic)

    return send
