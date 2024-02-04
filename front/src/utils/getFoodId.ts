interface FoodInfo {
  foodInfoId: string;
  foodName: string;
}

const getFoodId = (tag: string, foodInfo:FoodInfo[], searchInput:string) => {
  for (let i = 0; i < foodInfo.length; i++){
    if (foodInfo[i].foodName.split('_')[1] === tag || foodInfo[i].foodName.split('_')[1] ===(tag.split(searchInput)[0])) return foodInfo[i].foodInfoId;
    if (foodInfo[i].foodName===tag) return foodInfo[foodInfo.findIndex((info)=>info.foodName===tag)].foodInfoId;
  }
}

export default getFoodId;