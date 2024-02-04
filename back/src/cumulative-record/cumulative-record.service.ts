import { Injectable, NotFoundException } from "@nestjs/common";
import { CumulativeRecordRepository } from "./cumulative.repository";
import {
  CumulativeDateMealTypeDto,
  CumulativeRecordDateDto,
} from "./dto/cumulative-record.dto";
import { plainToInstance } from "class-transformer";
import { DataSource } from "typeorm";
import { HealthInfoRepository } from "../user/repositories/health-info.repository";
import { ImageRepository } from "src/image/repositories/image.repository";
import { CumulativeRecord } from "./cumulative-record.entity";
import { recommendIntakeData } from "./util/recommendIntake";

@Injectable()
export class CumulativeRecordService {
  constructor(
    private readonly cumulativeRepository: CumulativeRecordRepository,
    private readonly healthInfoRepository: HealthInfoRepository,
    private readonly imageRepository: ImageRepository,
    private readonly dataSource: DataSource
  ) {}

  async getDateRecord(date: Date, userId: string) {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      // [Cumulative Table] - 1) totalCalories, 2) totalNutrient
      const totalResult = await this.cumulativeRepository.getDateRecord(
        date,
        userId,
        queryRunner.manager
      );

      const totalResultDto = plainToInstance(
        CumulativeRecordDateDto,
        totalResult
      );
      // [HealthInfo Table] - 3) targetCalories, 4) recommendNutrient
      const HealthInfoResult =
        await this.healthInfoRepository.findHealthInfoByUserId(
          date,
          userId,
          "DESC",
          queryRunner.manager
        );
      if (!totalResultDto && !HealthInfoResult) {
        const HealthInfoResultASC =
          await this.healthInfoRepository.findHealthInfoByUserId(
            date,
            userId,
            "ASC",
            queryRunner.manager
          );
        if (HealthInfoResultASC) {
          const { targetCalories, recommendIntake } = HealthInfoResultASC;
          const result = recommendIntakeData(targetCalories, recommendIntake);
          await queryRunner.commitTransaction();
          return result;
        } else {
          const recommendNutrient = {
            carbohydrates: 10,
            proteins: 10,
            fats: 10,
            dietaryFiber: 10,
          };

          const result = {
            targetCalories: 10,
            recommendNutrient,
          };
          await queryRunner.commitTransaction();
          return result;
        }
      } else if (totalResultDto && !HealthInfoResult) {
        throw new NotFoundException("데이터 조회에서 오류가 발생했습니다");
      } else if (!totalResultDto && HealthInfoResult) {
        const { targetCalories, recommendIntake } = HealthInfoResult;
        const result = recommendIntakeData(targetCalories, recommendIntake);
        await queryRunner.commitTransaction();
        return result;
      }
      const { totalCalories, ...datas } = totalResultDto;
      const { targetCalories, recommendIntake } = HealthInfoResult;
      const { mealTypeResult, mealTypeImage } =
        await this.getDateMealTypeRecord(date, userId);
      const { recommendNutrient } = recommendIntakeData(
        targetCalories,
        recommendIntake
      );
      await queryRunner.commitTransaction();

      const dateArr = mealTypeResult.map((result, index) => [
        result.mealType,
        result.mealTotalCalories / 100,
        mealTypeImage[index],
      ]);
      const result = {
        totalCalories,
        targetCalories,
        totalNutrient: datas,
        recommendNutrient,
        dateArr,
      };
      return result;
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }

  async getDateMealTypeRecord(date: Date, userId: string) {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      const mealTypeResult =
        await this.cumulativeRepository.getDateMealTypeRecord(
          date,
          userId,
          queryRunner.manager
        );
      if (mealTypeResult.length === 0) {
        return mealTypeResult;
      }
      const mealTypeImage = [];
      mealTypeResult.map(async (image, index) => {
        const imageId = image.imageId;
        if (imageId) {
          const mealTypeImageResult =
            await this.imageRepository.getImageByImageId(
              String(imageId),
              queryRunner.manager
            );
          if(mealTypeImageResult){
            mealTypeImage[index] = mealTypeImageResult.foodImageUrl;
          }
        } else {
          mealTypeImage[index] = null;
        }
      });

      await queryRunner.commitTransaction();
      const result = {
        mealTypeResult,
        mealTypeImage,
      };
      return result;
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }

  async transformMealTypeRecord(date: Date, userId: string) {
    try {
      const { mealTypeResult, mealTypeImage } =
        await this.getDateMealTypeRecord(date, userId);
      if (!mealTypeResult || !mealTypeImage) {
        return [];
      }
      const dateArr = mealTypeResult.map((result, index) => [
        result.mealType,
        result.mealTotalCalories / 100,
        mealTypeImage[index],
      ]);
      const includeArr = mealTypeResult.map((item) => item.mealType);
      for (let i = 1; i <= 4; i++) {
        if (!includeArr.includes(i)) {
          dateArr.push([i, 0, null]);
        }
      }
      const sortedDateArr = dateArr.sort((a, b) => a[0] - b[0]);
      return sortedDateArr;
    } catch (error) {
      return error;
    }
  }

  async getMonthRecord(
    month: Date,
    userId: string
  ): Promise<CumulativeDateMealTypeDto[]> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();

    try {
      const result = await this.cumulativeRepository.getMonthRecord(
        month,
        userId,
        queryRunner.manager
      );
      await queryRunner.commitTransaction();
      return plainToInstance(CumulativeDateMealTypeDto, result);
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }

  async getMonthDetailRecord(month: Date, page: number, userId: string) {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const mealData = await this.cumulativeRepository.getMonthDetailRecord(
        month,
        page,
        userId,
        queryRunner.manager
      );

      const mealDataDto = plainToInstance(CumulativeRecord, mealData);
      const mealTypeImage = [];
      mealDataDto.map(async (image, index) => {
        const imageId = image.imageId;
        if (imageId) {
          const mealTypeImageResult =
            await this.imageRepository.getImageByImageId(
              String(imageId),
              queryRunner.manager
            );
          if(mealTypeImageResult){
            mealTypeImage[index] = mealTypeImageResult.foodImageUrl;
          }
        } else {
          mealTypeImage[index] = null;
        }
      });
      await queryRunner.commitTransaction();

      const groupedData = new Map<number, any[]>();

      mealDataDto.forEach((item, index) => {
        const dateKey = parseInt(new Date(item.date).getDate().toString());
        if (!groupedData.has(dateKey)) {
          groupedData.set(dateKey, []);
        }
        groupedData
          .get(dateKey)
          .push([
            item.mealType,
            item.mealTotalCalories / 100,
            mealTypeImage[index],
          ]);
      });
      const transformedData = [];
      groupedData.forEach((dateArr, date) => {
        transformedData.push({
          date,
          dateArr,
        });
      });

      return transformedData;
    } catch (error) {
      await queryRunner.rollbackTransaction();
      return error;
    } finally {
      await queryRunner.release();
    }
  }
}
