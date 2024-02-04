interface QuestionData {
  [key: string]: {
    type: {
      questionType: string;
      question: string | undefined;
    };
    text: string;
    button: { type: string; text: string }[];
  };
}

const questionData: QuestionData = {
  '1': {
    type: {
      questionType: '질문선택',
      question: undefined,
    },
    text: `안녕하세요, 저는 식단 관리를 함께할 AI봇 구그래미입니다!(●'◡'●)\n오늘 뭘 먹지 고민될 때는 식단을 추천드리고, 혼자서 식단 관리하기 어려울 때는 식단 평가와 목표 추천도 해드려요.\n도움이 필요하신 내용을 알려주세요!`,
    button: [
      { type: 'follow-up', text: '오늘 식단 추천이 필요해!' },
      { type: 'follow-up', text: '잘 하고 있는지, 식단 평가가 필요해!' },
      { type: 'ai', text: '나에게 맞는 목표를 추천해줄래?' },
    ],
  },
  '1-1': {
    type: {
      questionType: '식단추천',
      question: undefined,
    },
    text: `오늘 식단 추천이 필요하시군요! ‹( '▿' )›\n설정한 목표에 맞게 추천드릴 수도 있고 오늘 하루정도 치팅하고 싶다면 맛있는 식단도 추천 가능합니다.\n어떤 식단을 추천드릴까요?`,
    button: [
      { type: 'ai', text: '내 목표에 맞게 추천받고 싶어!' },
      { type: 'ai', text: '오늘은 맛있는 걸로 추천받을래!' },
    ],
  },
  '1-1-1': {
    type: {
      questionType: '식단추천',
      question: '내 목표에 맞게 추천받고 싶어!',
    },
    text: `추천받은 식단, 어떤가요?`,
    button: [
      { type: 'navigate', text: '추천받은 식단 기록하러 갈래!' },
      { type: 'follow-up', text: '다른 질문도 할래!' },
    ],
  },
  '1-1-2': {
    type: {
      questionType: '식단추천',
      question: '오늘은 맛있는 걸로 추천받을래!',
    },
    text: `추천받은 식단, 어떤가요?`,
    button: [
      { type: 'navigate', text: '추천받은 식단 기록하러 갈래!' },
      { type: 'follow-up', text: '다른 질문도 할래!' },
    ],
  },

  '1-2': {
    type: {
      questionType: '식단평가',
      question: undefined,
    },
    text: `평가를 시작할 준비가 되었습니다! ( ^ㅁ^ )ゞ\n객관적인 평가와 조언이 필요한 오늘 하루 혹은 이번주 식단관리 잘 하고 있는지 알려드릴게요!\n어떤 주기의 식단 평가가 필요하신가요?`,
    button: [
      { type: 'ai', text: '오늘 하루 내 식단은 어땠어?' },
      { type: 'ai', text: '이번주 식단 전체를 평가받을래!' },
    ],
  },
  '1-2-1': {
    type: {
      questionType: '식단평가',
      question: '오늘 하루 내 식단은 어땠어?',
    },

    text: `평가받은 내용, 어떤가요?`,
    button: [
      { type: 'navigate', text: '평가받은 식단 확인하러 갈래!' },
      { type: 'follow-up', text: '다른 질문도 할래!' },
    ],
  },
  '1-2-2': {
    type: {
      questionType: '식단평가',
      question: '이번주 식단 전체를 평가받을래!',
    },

    text: `평가받은 내용, 어떤가요?`,
    button: [
      { type: 'navigate', text: '평가받은 식단 확인하러 갈래!' },
      { type: 'follow-up', text: '다른 질문도 할래!' },
    ],
  },
  '1-3': {
    type: {
      questionType: '목표추천',
      question: '목표추천',
    },
    text: `식단관리를 위한 새로운 목표를 확인해보세요! ദ്ദി ˃ᴗ˂ )\n추천받은 목표, 어떤가요?`,
    button: [
      { type: 'navigate', text: '추천받은 목표 설정하러 갈래!' },
      { type: 'follow-up', text: '다른 질문도 할래!' },
    ],
  },
};

export { questionData };
