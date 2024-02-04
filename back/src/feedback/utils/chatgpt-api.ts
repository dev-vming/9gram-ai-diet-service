import { CumulativeRecordDateDto } from "src/cumulative-record/dto/cumulative-record.dto";

const dotenv = require("dotenv");
const { OpenAI } = require("openai");

dotenv.config();

const openai = new OpenAI();

export async function ChatgptApi(
  totalResult: CumulativeRecordDateDto,
  userInfo,
  questionType: string
) {
  const gender = (userInfo.gender = 1
    ? "남자야"
    : (userInfo.gender = 2 ? "여자야" : "사람이야"));

  const userInfoDetail = `유저는 키가 ${userInfo.height}cm이고 몸무게가 ${userInfo.weight}kg인 ${gender}`;

  let questionDetail = "";
  switch (questionType) {
    case "식단평가":
      questionDetail = `내가 오늘 하루 동안 먹은 식단의 영양성분은 탄수화물 ${totalResult.carbohydrates}g, 단백질 ${totalResult.proteins}g, 지방 ${totalResult.fats}g, 식이섬유 ${totalResult.dietaryFiber}g이야. 내 식단의 영양성분 구성을 평가해줘.`;
    case "식단추천":
      questionDetail = `나의 현재 몸무게는 ${userInfo[0].weight}kg이고 나는 ${userInfo[0].targetWeight}kg까지 몸무게를 빼고싶어. 다이어트 하기에 좋은 식단을 추천해줘`;
    case "목표추천":
      questionDetail = `나의 현재 몸무게는 ${userInfo[0].weight}kg이고 나는 ${userInfo[0].targetWeight}kg까지 몸무게를 빼고싶어. 나의 식단 기록 목표를 추천해줘`;
  }

  const chatCompletion = await openai.chat.completions.create({
    messages: [
      {
        role: "system",
        content:
          "너는 영양사야. 식단 영양성분 구성을 알려주면 1일 권장 섭취량을 기준으로 식단을 평가 해줘",
      },
      {
        role: "assistant",
        content: userInfoDetail,
      },
      {
        role: "user",
        content: questionDetail,
      },
    ],
    model: "gpt-3.5-turbo",
  });

  const outputText = chatCompletion.choices[0].message.content;
  return outputText;
}
