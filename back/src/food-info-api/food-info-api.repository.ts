import { EntityManager } from "typeorm";
import { FoodInfoApi } from "./food-info-api.entity";

export class FoodInfoAPIRepository {
  async saveDataPortal(data: string, manager: EntityManager) {
    const count = data.length;
    let message = false;
    for (let i = 0; i < count; i++) {
      const foodInfo = data[i];
      const foodInfoId = foodInfo["foodCd"];
      const foodName = foodInfo["foodNm"];
      const category = foodInfo["foodLv3Nm"];
      const mainCode = foodInfo["foodLv4Cd"];
      const mainFoodName = foodInfo["foodLv4Nm"];
      const brand =
        foodInfo["restNm"] !== "해당없음" ? foodInfo["restNm"] : null;
      const standardCapacity =
        foodInfo["nutConSrtrQua"] !== ""
          ? Math.round(
              parseFloat(foodInfo["nutConSrtrQua"].replace(/[^0-9]/g, "")) * 100
            )
          : null;
      const totalCapacity =
        foodInfo["foodSize"] !== ""
          ? Math.round(
              parseFloat(foodInfo["foodSize"].replace(/[^0-9]/g, "")) * 100
            )
          : null;
      const calories =
        foodInfo["enerc"] !== ""
          ? Math.round(parseFloat(foodInfo["enerc"]) * 100)
          : null;
      const carbohydrates =
        foodInfo["chocdf"] !== ""
          ? Math.round(parseFloat(foodInfo["chocdf"]) * 100)
          : null;
      const proteins =
        foodInfo["prot"] !== ""
          ? Math.round(parseFloat(foodInfo["prot"]) * 100)
          : null;
      const fats =
        foodInfo["fatce"] !== ""
          ? Math.round(parseFloat(foodInfo["fatce"]) * 100)
          : null;
      const dietaryFiber =
        foodInfo["fibtg"] !== ""
          ? Math.round(parseFloat(foodInfo["fibtg"]) * 100)
          : null;
      const createdDate = foodInfo["crtYmd"];
      const updatedDate = foodInfo["crtrYmd"];

      await manager.save(FoodInfoApi, {
        foodInfoId,
        foodName,
        category,
        mainCode,
        mainFoodName,
        brand,
        standardCapacity,
        totalCapacity,
        calories,
        carbohydrates,
        proteins,
        fats,
        dietaryFiber,
        createdDate,
        updatedDate,
      });
      message = true;
    }
    return message;
  }

  async saveDataPortalProcess(data: string, manager: EntityManager) {
    const count = data.length;
    let message = false;
    for (let i = 0; i < count; i++) {
      const foodInfo = data[i];
      const foodInfoId = foodInfo["foodCd"];
      const foodName = foodInfo["foodNm"];
      const category = foodInfo["foodLv3Nm"];
      const mainCode = foodInfo["foodLv4Cd"];
      const mainFoodName = foodInfo["foodLv4Nm"];
      const brand = foodInfo["mfrNm"] !== "해당없음" ? foodInfo["mfrNm"] : null;
      const standardCapacity =
        foodInfo["nutConSrtrQua"] !== ""
          ? Math.round(
              parseFloat(foodInfo["nutConSrtrQua"].replace(/[^0-9]/g, "")) * 100
            )
          : null;
      const totalCapacity =
        foodInfo["foodSize"] !== ""
          ? Math.round(
              parseFloat(foodInfo["foodSize"].replace(/[^0-9]/g, "")) * 100
            )
          : null;
      const calories =
        foodInfo["enerc"] !== ""
          ? Math.round(parseFloat(foodInfo["enerc"]) * 100)
          : null;
      const carbohydrates =
        foodInfo["chocdf"] !== ""
          ? Math.round(parseFloat(foodInfo["chocdf"]) * 100)
          : null;
      const proteins =
        foodInfo["prot"] !== ""
          ? Math.round(parseFloat(foodInfo["prot"]) * 100)
          : null;
      const fats =
        foodInfo["fatce"] !== ""
          ? Math.round(parseFloat(foodInfo["fatce"]) * 100)
          : null;
      const dietaryFiber =
        foodInfo["fibtg"] !== ""
          ? Math.round(parseFloat(foodInfo["fibtg"]) * 100)
          : null;
      const createdDate = foodInfo["crtYmd"];
      const updatedDate = foodInfo["crtrYmd"];

      const result = await manager.create(FoodInfoApi, {
        foodInfoId,
        foodName,
        category,
        mainCode,
        mainFoodName,
        brand,
        standardCapacity,
        totalCapacity,
        calories,
        carbohydrates,
        proteins,
        fats,
        dietaryFiber,
        createdDate,
        updatedDate,
      });

      await manager.save(result);
      message = true;
    }
    return message;
  }
}
