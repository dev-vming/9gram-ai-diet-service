import { Injectable } from "@nestjs/common";
import { FoodInfoRepository } from "./food-info.repository";
import { FoodInfoDto } from "./dto/food-info.dto";
import { plainToInstance } from "class-transformer";
import { DataSource } from "typeorm";

@Injectable()
export class FoodInfoService {
  constructor(
    private readonly foodInfoRepository: FoodInfoRepository,
    private readonly dataSource: DataSource
  ) {}

  async getFoodList(keyword: string): Promise<FoodInfoDto[]> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      const result = await this.foodInfoRepository.getFoodList(
        keyword,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      return result;
    } catch (error) {
      await queryRunner.rollbackTransaction();
    } finally {
      await queryRunner.release();
    }
  }

  async getFoodNextList(
    keyword: string,
    lastFoodId: string
  ): Promise<FoodInfoDto[]> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      const result = await this.foodInfoRepository.getFoodNextList(
        keyword,
        lastFoodId,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      return result;
    } catch (error) {
      await queryRunner.rollbackTransaction();
    } finally {
      await queryRunner.release();
    }
  }

  async getFoodInfo(foodName: string): Promise<FoodInfoDto> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      const result = await this.foodInfoRepository.getFoodInfo(
        foodName,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      return plainToInstance(FoodInfoDto, result);
    } catch (error) {
      await queryRunner.rollbackTransaction();
    } finally {
      await queryRunner.release();
    }
  }

  async getFoodInfoById(foodInfoId: string): Promise<FoodInfoDto> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      const result = await this.foodInfoRepository.getFoodInfoById(
        foodInfoId,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      return plainToInstance(FoodInfoDto, result);
    } catch (error) {
      await queryRunner.rollbackTransaction();
    } finally {
      await queryRunner.release();
    }
  }

  async getFoodIdList(foodList: string[]) {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      const foodInfoIdList = [];
      for (const food of foodList) {
        const result = await this.foodInfoRepository.getFoodInfo(
          food,
          queryRunner.manager
        );
        foodInfoIdList.push(result.foodInfoId);
      }
      await queryRunner.commitTransaction();
      return foodInfoIdList;
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }
}
