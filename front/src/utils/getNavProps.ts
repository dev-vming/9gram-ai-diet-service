import { TopBarPropsType, TopNavKeyType } from 'typings/propTypes';

export const existDate = (val: string) => {
  const dateRegExp = /(\d{4}-\d{1,2}-\d{1,2})/;
  const dateInUrl = val.match(dateRegExp);
  return {
    existed: !!(dateInUrl && dateInUrl[1]),
    date: dateInUrl ? dateInUrl[1] : null,
  };
};

//url[2]의 숫자를 meal로 변경
export const getKeyFromUrl = (url: string): TopNavKeyType | string => {
  //url에 date가 존재하는 경우 ymd로 변경
  const urlArr = url.split('/');
  const returnKey: TopNavKeyType | string = urlArr
    .map((url, idx) => {
      const deleteHypen = url.replace(/-/g, '');
      const dateExisted = existDate(url).existed;
      if (dateExisted) {
        return 'ymd';
      }
      if (idx === 1 && Number(url)) {
        return 'step';
      }
      if (idx === 2) {
        return 'meal';
      }
      if (urlArr[idx - 1] === 'share') {
        return 'id';
      }
      return deleteHypen;
    })
    .join('');
  return returnKey;
};

export const defaultNavProps: TopBarPropsType = {
  home: false,
  title: undefined,
  back: false,
  qIcon: false,
  icon: false,
};

interface GetNavPropsType {
  [key: string]: {};
}

export const getNavProps: GetNavPropsType = {
  join: { title: '회원정보 입력', back: true },
  onboardingstep: { back: true },
  login: { back: true, title: '로그인' },
  home: { home: true, title: '구그램' },
  calendar: { title: '식단달력' },
  recordymd: { title: '하루식단' },
  recordymdmeal: {
    title: 'AI 식단 기록',
    back: true,
    qIcon: true,
    //뒤로가기 할ㅐ selectedDate로
  },
  addphotoymdmeal: { title: 'AI 식단 카메라', back: true, qIcon: true },
  recordymdmealedit: { title: 'AI 식단 확인', back: true },
  addphotoymdmealsearch: { title: '식단 검색', back: true },
  aianalyze: { title: 'AI 분석', icon: true },
  aidrawer: { back: true, title: 'AI 분석 서랍' },
  aidrawerdetail: { back: true, title: '' },
  shareid: { home: true, title: '구그램' },
  mypage: { title: '설정' },
  mypageedit: { back: true, title: '정보수정' },
};
