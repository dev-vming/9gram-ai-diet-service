import { HttpException, Injectable } from "@nestjs/common";
import { FoodInfo } from "./food-info.entity";
import { EntityManager } from "typeorm";
import { FoodInfoDto } from "./dto/food-info.dto";

@Injectable()
export class FoodInfoRepository {
  async getFoodList(
    keyword: string,
    manager: EntityManager
  ): Promise<FoodInfo[]> {
    try {
      return await manager
        .createQueryBuilder(FoodInfo, "entity")
        .select(["entity.foodInfoId", "entity.foodName"])
        .where("REPLACE(entity.food_name, ' ', '') like :keyword", {
          keyword: `%${keyword}%`,
        })
        .orderBy("entity.food_name", "ASC")
        .take(10)
        .getMany();
    } catch (error) {
      throw new HttpException(error.detail, 500);
    }
  }

  async getFoodNextList(
    keyword: string,
    lastFoodId: string,
    manager: EntityManager
  ): Promise<FoodInfo[]> {
    try {
      const result = await manager
        .createQueryBuilder(FoodInfo, "entity")
        .select(["entity.foodInfoId", "entity.foodName"])
        .where("REPLACE(entity.food_name, ' ', '') like :keyword", {
          keyword: `%${keyword}%`,
        })
        .andWhere("entity.food_info_id > :lastFoodId", { lastFoodId })
        .orderBy("entity.food_info_id", "ASC")
        .take(10)
        .getMany();
      return result;
    } catch (error) {
      throw new HttpException(error.detail, 500);
    }
  }

  async getFoodInfo(
    foodName: string,
    manager: EntityManager
  ): Promise<FoodInfoDto> {
    try {
      const firstResult = await manager
        .createQueryBuilder(FoodInfo, "entity")
        .where("REPLACE(entity.food_name, ' ', '') = :foodName", { foodName })
        .orderBy("entity.created_date", "DESC")
        .getOne();
      if (!firstResult) {
        const result = await manager
          .createQueryBuilder(FoodInfo, "entity")
          .where("REPLACE(entity.food_name, ' ', '') like :foodName", {
            foodName: `%${foodName}%`,
          })
          .orderBy("entity.created_date", "DESC")
          .getOne();
        return result;
      }
      return firstResult;
    } catch (error) {
      throw error;
    }
  }

  async getFoodInfoById(
    foodInfoId: string,
    manager: EntityManager
  ): Promise<FoodInfoDto> {
    try {
      return await manager
        .createQueryBuilder(FoodInfo, "entity")
        .where("entity.food_info_id = :foodInfoId", { foodInfoId })
        .getOneOrFail();
      //getOneOrFail: 결과값 없을 경우 EntityNotFoundError 던짐
    } catch (error) {
      throw error;
    }
  }
}
