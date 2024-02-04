import { ArgumentMetadata, Injectable, PipeTransform } from "@nestjs/common";

@Injectable()
export class FoodNamePipe implements PipeTransform {
  transform(foodName: any, metadata: ArgumentMetadata) {
    // 다른 방안 고민해보기
    if (Array.isArray(foodName)) {
      const foodList = [];
      for (const food of foodName) {
        const result = foodNameCase(food.replace(" ", ""));
        foodList.push(result);
      }
      return foodList;
    } else {
      return foodNameCase(foodName);
    }
  }
}

const foodNameCase = (foodName: string) => {
  if (foodName) {
    let food = foodName.replace(" ", "");
    switch (food) {
      case "기타잡곡밥":
        food = "잡곡밥";
        break;
      case "콩밥":
        food = "콩밥_검정콩";
        break;
      case "참치김밥":
        food = "김밥_참치";
        break;
      case "오일소스스파게티":
        food = "스파게티_오일소스";
        break;
      case "크림소스스파게티":
        food = "스파게티_크림소스";
        break;
      case "배추된장국":
        food = "된장국_배추";
        break;
      case "치킨윙":
        food = "햄_스모크치킨윙";
        break;
      case "소세지볶음":
        food = "소시지볶음";
        break;
      default:
        food;
    }
    return food;
  }
};
